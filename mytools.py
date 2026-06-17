import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import random
import os
import math
from configs import get_my_config, parse_args

class ConfigObject:
    """将字典转换为对象的包装类"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
def convert_configs(configs):
    """检查configs是否为字典，如果是则转换为对象"""
    if isinstance(configs, dict):
        return ConfigObject(configs)
    return configs

def set_all_seeds(seed):
    """
    设置所有随机种子以确保结果可复现
    """
    # 系统库
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时

    # 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def unpatchify_with_batch(patched_tensor, patch_size, original_channels):
    """
    对带 Batch 的 patchified tensor 进行还原，恢复为原始 (B, T, C, H, W) 格式。

    参数:
        patched_tensor: torch.Tensor，形状为 (B, T, C * p * p, H', W')
        patch_size: int，patch 大小 p
        original_channels: int，原始通道数 C

    返回:
        torch.Tensor，形状为 (B, T, C, H, W)
    """
    B, T, Cp2, H_, W_ = patched_tensor.shape
    p = patch_size
    C = original_channels

    assert Cp2 == C * p * p, f"通道数不匹配：{Cp2} != {C} * {p} * {p}"

    # 恢复 patch 通道维度为 patch 格式
    x = patched_tensor.reshape(B, T, C, p, p, H_, W_)

    # 交换维度：将 patch 中像素恢复到空间位置
    x = x.permute(0, 1, 2, 5, 3, 6, 4)  # (B, T, C, H', p, W', p)

    # 合并 patch 像素还原空间维度
    x = x.reshape(B, T, C, H_ * p, W_ * p)

    return x


class MSELossIgnoreNaN(nn.Module):
    def __init__(self, config, mask_valid: torch.Tensor=None, patched=False):
        """
        :param mask_valid: valid: True, invalid: False
        :param patched: if the pred and target are patched
        """
        super().__init__()
        self.mse_func = nn.MSELoss(reduction="sum")

        if mask_valid is not None:
            self.mask_valid = mask_valid[None,None,None,:,:].to(device=config.device)
            H, W = mask_valid.shape
            print(f"mask_valid: {self.mask_valid.shape}")
            if patched:
                C = config.output_channels
                p = config.patch_size
                self.mask_valid = self.mask_valid.expand(1, 1, C, H, W)
                self.mask_valid = self.mask_valid.reshape(C, H // p, p, W // p, p)
                self.mask_valid = self.mask_valid.permute(0, 2, 4, 1, 3).reshape(1, 1, C * p * p, H // p, W // p)
                print(f"after patched: {self.mask_valid.shape}")

    def forward(self, pred, target):
        if hasattr(self, "mask_valid"):
            mask_valid = self.mask_valid.expand_as(pred)
        else:
            mask_valid = ~(torch.isnan(target) | torch.isinf(target))

        valid_count = mask_valid.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        target = torch.where(mask_valid, target, pred)
        loss = self.mse_func(pred, target) / valid_count

        return loss

class MaskPearsonCorr(nn.Module):
    def __init__(self, mask_valid: torch.Tensor = None):
        """
        :param mask_valid: valid: True, invalid: False
        """
        super().__init__()
        self.mask_valid = mask_valid

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_pvalue: bool = False):
        """
        计算 pred 和 target 之间的整体 Pearson 相关系数，忽略 target 中为 NaN 的位置。

        参数:
            pred (torch.Tensor): 预测值张量，形状为 [M, N]
            target (torch.Tensor): 目标张量，形状为 [M, N]
            return_pvalue (bool): 是否同时返回 p 值

        返回:
            corr (torch.Tensor): 标量相关系数
            [可选] p_value (torch.Tensor): 标量 p 值
        """
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")

        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # 掩膜：忽略 target 中为 NaN 的位置
        if self.mask_valid is not None:
            mask = self.mask_valid.view(-1)
        else:
            mask = ~torch.isnan(target_flat)

        pred_masked = pred_flat[mask]
        target_masked = target_flat[mask]

        n = pred_masked.numel()
        if n == 0:
            nan_tensor = torch.tensor(float('nan'), device=pred.device)
            return (nan_tensor, nan_tensor) if return_pvalue else nan_tensor

        # 计算 Pearson 相关
        pred_mean = pred_masked.mean()
        target_mean = target_masked.mean()

        pred_centered = pred_masked - pred_mean
        target_centered = target_masked - target_mean

        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2))

        epsilon = 1e-8  # 防止除以0
        corr = numerator / (denominator + epsilon)

        # 截断到 [-1, 1] 避免数值精度引起的轻微越界
        corr = torch.clamp(corr, -1.0, 1.0)

        if not return_pvalue:
            return corr

        if n < 3:
            return corr, torch.tensor(float('nan'), device=pred.device)

        # 计算 P 值: t 统计量。
        # 由于物理/图像数据有效点通常极大 (n > 30)，t 分布极其逼近标准正态分布
        t_stat = corr * torch.sqrt((n - 2) / (1 - corr ** 2 + epsilon))
        # 使用误差函数 (erf) 高效计算正态分布的双侧 P 值
        p_value = 2 * (1 - 0.5 * (1 + torch.erf(torch.abs(t_stat) / math.sqrt(2))))

        return corr, p_value


class MaskPearsonCorrNP:
    def __init__(self, mask_valid: np.ndarray = None):
        """
        :param mask_valid: valid: True, invalid: False
        """
        self.mask_valid = mask_valid

    def __call__(self, pred: np.ndarray, target: np.ndarray, return_pvalue: bool = False):
        """
        计算 pred 和 target 之间的整体 Pearson 相关系数，忽略 target 中为 NaN 的位置。

        参数:
            pred (np.ndarray): [M, N]
            target (np.ndarray): [M, N]
            return_pvalue (bool): 是否返回 p 值

        返回:
            corr: float
            [可选] p_value: float
        """
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")

        # flatten
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        # mask
        if self.mask_valid is not None:
            mask = self.mask_valid.reshape(-1)
        else:
            mask = ~np.isnan(target_flat)

        pred_masked = pred_flat[mask]
        target_masked = target_flat[mask]

        n = pred_masked.size
        if n == 0:
            if return_pvalue:
                return np.nan, np.nan
            return np.nan

        # mean
        pred_mean = pred_masked.mean()
        target_mean = target_masked.mean()

        # center
        pred_centered = pred_masked - pred_mean
        target_centered = target_masked - target_mean

        numerator = np.sum(pred_centered * target_centered)
        denominator = np.sqrt(
            np.sum(pred_centered ** 2) * np.sum(target_centered ** 2)
        )

        epsilon = 1e-8
        corr = numerator / (denominator + epsilon)

        # clamp
        corr = np.clip(corr, -1.0, 1.0)

        if not return_pvalue:
            return corr

        if n < 3:
            return corr, np.nan

        # t-stat
        t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2 + epsilon))

        # 使用 erf 近似正态分布
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

        return corr, p_value

def compute_gradients_exact(sla, lon, lat, R_E=6371000.0, is_real_gradient=True):
    """
    使用中心差分 (torch.gradient) 计算高精度物理梯度，比 Sobel 算子更符合物理场真实特征。
    输入 sla 尺寸为 [B, T, C, H, W]
    返回:
      grad_x_phys, grad_y_phys: 分别为沿 x (经度) 和 y (纬度) 方向的梯度
    """
    if lon.ndim == 2 and np.all(lon == lon[0, :][None, :]) and np.all(lat == lat[:, 0][:, None]):
        lon = lon[0, :]
        lat = lat[:, 0]

    B, T, C, H, W = sla.shape

    # dim=-1 是 W (经度 x), dim=-2 是 H (纬度 y)
    # torch.gradient 默认计算中心差分： (f(i+1) - f(i-1)) / 2
    grad_y = torch.gradient(sla, dim=-2)[0]
    grad_x = torch.gradient(sla, dim=-1)[0]

    if is_real_gradient:
        # 纬度间隔 (恒定)
        delta_lat = lat[1] - lat[0]
        dy = delta_lat * (np.pi / 180.0) * R_E

        # 经度间隔 (随纬度变化, H个值)
        delta_lon = lon[1] - lon[0]
        lat_rad = np.deg2rad(lat)
        dx_per_row = delta_lon * (np.pi / 180.0) * R_E * np.cos(lat_rad)

        # 转换为 Tensor 并广播到 [1, 1, 1, H, 1] 匹配 sla 的维度
        dx_tensor = torch.tensor(dx_per_row, dtype=sla.dtype, device=sla.device).view(1, 1, 1, H, 1)

        # 物理距离缩放
        grad_x_phys = grad_x / dx_tensor
        grad_y_phys = grad_y / dy

        return grad_x_phys, grad_y_phys
    else:
        return grad_x, grad_y

def compute_gradients_sobel(sla, lon, lat, R_E=6371000.0, is_real_gradient=True):
    """
    使用Sobel算子计算梯度，输入sla尺寸为 [B, T, C, H, W]，
    利用二维经纬度数据（lon, lat均为一维提取自二维数据）计算每行dx和全局dy。
    返回:
      grad_x_phys, grad_y_phys: 分别为沿x和y方向的物理梯度, 尺寸均为 [B, T, C, H, W]
    """
    if lon.ndim == 2 and np.all(lon == lon[0, :][None, :]) and np.all(lat == lat[:, 0][:, None]):
        lon = lon[0,:]
        lat = lat[:,0]
    B, T, C, H, W = sla.shape
    # 合并B和T维度进行卷积计算
    x = sla.reshape(B * T, C, H, W)  # [B*T, C, H, W]

    # 定义Sobel卷积核（Sobel算子）
    kernel_sobel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=x.dtype, device=x.device) / 8.0
    kernel_sobel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=x.dtype, device=x.device) / 8.0
    # 为每个通道创建独立的卷积核，使用grouped convolution实现通道分离
    kernel_sobel_x = kernel_sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    kernel_sobel_y = kernel_sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # 使用replicate模式的padding保证尺寸不变（对3×3核，padding=1）
    x_padded = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
    # 使用分组卷积，每个通道独立计算
    grad_x = F.conv2d(x_padded, kernel_sobel_x, groups=C)
    grad_y = F.conv2d(x_padded, kernel_sobel_y, groups=C)
    if is_real_gradient:
        # 经纬度差
        # 假设lon, lat为从二维经纬度数据中提取的一维数组
        delta_lat = lat[1] - lat[0]  # 单位：度
        delta_lon = lon[1] - lon[0]  # 单位：度
        dy = delta_lat * (np.pi / 180) * R_E  # m
        lat_rad = np.deg2rad(lat)  # [H]
        dx_per_row = delta_lon * (np.pi / 180) * R_E * np.cos(lat_rad)  # [H] todo:改成常数是否会变差？
        dx_tensor = torch.tensor(dx_per_row, dtype=x.dtype, device=x.device).view(1, 1, H, 1)

        grad_x_phys = grad_x / dx_tensor
        grad_y_phys = grad_y / dy

        # 恢复原始尺寸 [B, T, C, H, W]
        grad_x_phys = grad_x_phys.view(B, T, C, H, W)
        grad_y_phys = grad_y_phys.view(B, T, C, H, W)

        return grad_x_phys, grad_y_phys
    else:
        return grad_x, grad_y



def compute_f_and_sigmoid_weight(lat,k=2,phi0=5, if_solid_f=False):
    """
    基于纬度计算科氏参数 f 和 sigmoid 型的 f_weight

    参数:
        lat: numpy array, shape [H, W]，网格纬度

    返回:
        f: torch.Tensor, shape [1,1,1,H,W]，科氏参数
        f_weight: torch.Tensor, shape [1,1,1,H,W]，sigmoid 权重
    """
    Omega = 7.2921e-5  # 地球自转角速度
    lat_t = torch.from_numpy(lat).float()  # [H, W]
    phi = torch.abs(lat_t)

    if if_solid_f:
        f = 2 * Omega * torch.sin(torch.deg2rad(torch.mean(lat_t)))  # [H, W]
        f_weight = 1.0
        return f, f_weight

    # 空间变化的科氏参数
    f = 2 * Omega * torch.sin(torch.deg2rad(lat_t))  # [H, W]
    f = f[None, None, None, :, :]  # [1,1,1,H,W]

    f_weight = 1 / (1.0 + torch.exp(-k * (phi - phi0)))  # [H, W]
    f_weight = f_weight[None, None, None, :, :]  # [1,1,1,H,W]

    return f, f_weight

def compute_f_and_gaussian_weight(lat,theta=2.2):
    """

    参数:
        lat: numpy array, shape [H, W]，网格纬度

    返回:
        f: torch.Tensor, shape [1,1,1,H,W]，科氏参数
        f_weight: torch.Tensor, shape [1,1,1,H,W]，sigmoid 权重
    """
    Omega = 7.2921e-5  # 地球自转角速度
    lat_t = torch.from_numpy(lat).float()  # [H, W]
    phi = torch.abs(lat_t)

    # 空间变化的科氏参数
    f = 2 * Omega * torch.sin(torch.deg2rad(lat_t))  # [H, W]
    f = f[None, None, None, :, :]  # [1,1,1,H,W]

    f_weight = 1-torch.exp(-(phi/theta)**2)  # [H, W]
    f_weight = f_weight[None, None, None, :, :]  # [1,1,1,H,W]

    return f, f_weight

def compute_geostrophic_current(pred,  lon, lat, if_solid_f=False):
    """
    基于空间 f 计算地转流速度（物理单位 m/s）。
    """
    g = 9.81  # 重力加速度

    f, f_weight = compute_f_and_sigmoid_weight(lat, if_solid_f)
    f = f.to(pred.device)

    grad_x, grad_y = compute_gradients_sobel(pred, lon, lat, R_E=6.371e6) #todo
    # grad_x, grad_y = compute_gradients_exact(pred, lon, lat)
    # 地转流分量
    u_geo = - (g / f) * grad_y
    v_geo = (g / f) * grad_x

    return u_geo, v_geo, f_weight


def reverse_schedule_sampling(itr,  total_length, input_length, img_shape, args, reverse=True, mode='train'):
    """
    PyTorch版本 - 支持训练和测试的reverse schedule sampling

    Args:
        itr: 当前迭代步数
        total_length: 总序列长度
        input_length: 输入序列长度
        img_shape: 图像形状 (C, H, W)
        args: 参数对象
        mode: 模式 'train' 或 'test'
        :param reverse:
    """
    C, H, W = img_shape

    # 测试模式：完全使用真实帧（不进行mask）
    if mode == 'test':
        # 创建全为True的mask，表示所有帧都使用真实值
        real_input_flag = torch.ones(( total_length - 2, C, H, W),device=args.device)
        real_input_flag[input_length-1:] = 0

        return real_input_flag

    # 训练模式：根据调度策略生成mask
    elif mode == 'train':
        # 1. 计算调度概率
        r_eta_st, eta_st = 0.5, 0.5
        if reverse:
            if itr < args.r_sampling_step_1:
                r_eta, eta = r_eta_st, eta_st
                print(f"r_eta: {r_eta}, eta: {eta}")

            elif itr < args.r_sampling_step_2:

                r_eta =r_eta_st + (1-r_eta_st)*(1.0 -  math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha))
                eta = eta_st * (1 - ((itr - args.r_sampling_step_1) / (args.r_sampling_step_2 - args.r_sampling_step_1)))
                print(f"r_eta: {r_eta}, eta: {eta}")

            else:
                r_eta, eta = 1.0, 0.0
        else:
            if itr < args.r_sampling_step_1:
                r_eta, eta = r_eta_st, eta_st
                print(f"r_eta: {r_eta}, eta: {eta}")
            elif itr < args.r_sampling_step_2:
                r_eta = 1.0
                eta = eta_st - eta_st*(1 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
                print(f"r_eta: {r_eta}, eta: {eta}")

            else:
                r_eta, eta = 1.0, 0.0
                print(f"r_eta: {r_eta}, eta: {eta}")

        # 2. 使用torch.bernoulli生成mask
        # 输入序列内部的mask (t=1 到 input_length-1)
        r_mask = torch.bernoulli(
            torch.full(( input_length - 1, C, H, W), r_eta, device=args.device)
        )

        # 预测序列的mask (t=input_length 到 total_length-1)
        pred_mask = torch.bernoulli(
            torch.full(( total_length - input_length -1, C, H, W), eta, device=args.device)
        )

        # 3. 合并mask
        real_input_flag = torch.cat([r_mask, pred_mask], dim=0)

        return real_input_flag

    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'test'.")