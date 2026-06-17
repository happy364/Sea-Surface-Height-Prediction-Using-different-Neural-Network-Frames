import os
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
from plotter.spatial_plotter import PlotterMultiPanel
from mytools import MaskPearsonCorr, unpatchify_with_batch


# ==============================
# 向量化相关系数
# ==============================
def calc_temporal_corr_p_vectorized(x, y, axis=0):
    N = x.shape[axis]
    x_mean = np.mean(x, axis=axis, keepdims=True)
    y_mean = np.mean(y, axis=axis, keepdims=True)

    x_dev = x - x_mean
    y_dev = y - y_mean

    cov = np.sum(x_dev * y_dev, axis=axis)
    var_x = np.sum(x_dev ** 2, axis=axis)
    var_y = np.sum(y_dev ** 2, axis=axis)

    denominator = np.sqrt(var_x * var_y)
    corr = np.divide(cov, denominator, out=np.zeros_like(cov), where=denominator != 0)

    df = N - 2
    r_clipped = np.clip(corr, -1.0 + 1e-8, 1.0 - 1e-8)
    t_stat = r_clipped * np.sqrt(df / (1.0 - r_clipped ** 2))
    p_val = 2 * stats.t.sf(np.abs(t_stat), df)

    return corr, p_val


# ==============================
# 主分析类
# ==============================
class WindStressAnalyzer:
    def __init__(self, args, dataset, pearson_tool, model_A, model_B=None):
        self.args = args
        self.dataset = dataset
        self.model_A = model_A
        self.model_B = model_B
        self.device = args.device
        self.pearson_tool = pearson_tool

    def from_input_data(self, input_data):
        """
        Returns:
            inputs: Tensor
            targets: Tensor or None
        """

        if isinstance(input_data, (tuple, list)):
            inputs, targets = input_data

        elif isinstance(input_data, torch.Tensor):
            inputs = input_data

            if inputs.size(1) > self.args.input_length:
                targets = inputs[:, self.args.input_length:]
            else:
                targets = None
        else:
            raise ValueError("input_data must be tuple, list or torch.Tensor")

        inputs = inputs.to(self.device)

        if targets is not None:
            targets = targets.to(self.device)

        return inputs, targets

    def run_full_analysis_time(self, save_dir, mask_land):
        os.makedirs(save_dir, exist_ok=True)
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False, num_workers=4)

        # 移除了 curl，只保留 wind
        collected = {'err': [], 'wind': [], 'imp': []}

        self.model_A.model.eval()
        if self.model_B:
            self.model_B.model.eval()

        print("Stage 1: Collecting data (Wind Speed only)...")
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                datax, datay = self.from_input_data(data)
                if self.args.patched:
                    inputs = datax[:, :, :self.args.patch_size **2]
                else:
                    inputs = datax[:, :, :-2]
                pred_A = self.model_A.predict(inputs)
                if self.args.patched:
                    datay = unpatchify_with_batch(
                        datay,
                        self.args.patch_size,
                        self.args.output_channels +2
                    )
                err = torch.abs(pred_A[:, :, 0] - datay[:, :, 0])

                imp = None
                if self.model_B:
                    pred_B = self.model_B.predict(inputs)
                    err_B = torch.abs(pred_B[:, :, 0] - datay[:, :, 0])
                    imp = err - err_B

                # 只计算风速大小
                u, v = datay[:, :, -2], datay[:, :, -1]
                wind_mag = torch.sqrt(u ** 2 + v ** 2)

                collected['err'].append(err.cpu().numpy())
                collected['wind'].append(wind_mag.cpu().numpy())
                if imp is not None:
                    collected['imp'].append(imp.cpu().numpy())

        # 拼接
        err = np.concatenate(collected['err'], axis=0)
        wind = np.concatenate(collected['wind'], axis=0)
        imp = np.concatenate(collected['imp'], axis=0) if collected['imp'] else None

        print("Stage 2: Calculating Correlation & Significance...")
        corr_wind_err, p_wind_err = calc_temporal_corr_p_vectorized(wind, err)

        corr_wind_imp, p_wind_imp = None, None
        if imp is not None:
            corr_wind_imp, p_wind_imp = calc_temporal_corr_p_vectorized(wind, imp)

        # ==============================
        # 准备绘图数据
        # ==============================
        print("Stage 3: Multi-panel plotting with significance hatches...")
        lon_min, lon_max = np.min(self.dataset.lon), np.max(self.dataset.lon)
        lat_min, lat_max = np.min(self.dataset.lat), np.max(self.dataset.lat)
        extent = [lon_min, lon_max, lat_min, lat_max]

        plotter = PlotterMultiPanel(self.dataset.lon, self.dataset.lat, extent=extent)
        mask_valid = ~(mask_land.cpu().numpy() if torch.is_tensor(mask_land) else mask_land)

        target_leads = [0, 3, 6, 9]
        valid_leads = [t for t in target_leads if t < err.shape[1]]

        def prep_with_significance(corr, p):
            corr_maps = []
            sig_masks = []  # 用于存显著性区域（True表示显著，将画斜线）
            for t in valid_leads:
                # 保持所有点的值，仅掩盖陆地
                c_map = np.where(mask_valid, corr[t], np.nan)
                # 显著性掩码：p < 0.05 且在海上的点
                s_mask = (p[t] < 0.05) & mask_valid

                corr_maps.append(c_map)
                sig_masks.append(s_mask)
            return corr_maps, sig_masks

        # 组织数据结构
        # 注意：这里假设你的 PlotterMultiPanel.plot_panel 能接收显著性掩码
        # 如果 plotter 不直接支持，你可能需要修改 plotter.py 里的 ax.contourf 部分
        plot_data = {}
        sig_data = {}

        c_maps, s_masks = prep_with_significance(corr_wind_err, p_wind_err)
        plot_data["Err-Wind"] = c_maps
        sig_data["Err-Wind"] = s_masks

        if imp is not None:
            c_maps_imp, s_masks_imp = prep_with_significance(corr_wind_imp, p_wind_imp)
            plot_data["Imp-Wind"] = c_maps_imp
            sig_data["Imp-Wind"] = s_masks_imp

        # 计算 colorbar 范围
        # all_vals = np.concatenate([np.ravel(v) for v in plot_data.values()])
        # vmax = np.nanmax(np.abs(all_vals)) if np.any(~np.isnan(all_vals)) else 1.0
        vmax = 0.62

        save_path = os.path.join(save_dir, "wind_corr_significance.png")

        # 核心变动：传入 sig_data 并在 plotter 中处理 hatch
        plotter.plot_panel(
            data=plot_data,
            significance_data=sig_data,  # 传递显著性掩码
            cbar_range={'data': (-vmax, vmax)},
            title="Correlation with Wind Speed (Hatches indicate p < 0.05)",
            save_path=save_path,
            cmap='RdBu_r',
            hatch='////',  # 指定斜线样式
        )

        print(f"Saved: {save_path}")

# ==============================
# main
# ==============================
if __name__ == "__main__":
    from configs import parse_args, get_my_config
    from dataset import MvDataset
    from tools.base_method import Model
    from mytools import set_all_seeds
    import numpy as np
    from models import SimVP_Model, RNN

    args_ = parse_args()
    args_.need_wind = True
    args_.model_name = 'predrnn'
    args = get_my_config(args_)

    set_all_seeds(args.SEED)

    mask_land = torch.from_numpy(np.load(args.path_land_mask))
    stds = np.load(args.path_stds)

    #
    # modelA = SimVP_Model(**args.model_config)
    # modelB = SimVP_Model(**args.model_config)
    #
    # model_para_path_A = r"/data/hjj/ssh_prediction/work_dir/scs/parameter_3b_gc_0/pinn0_b4/SV_20260103_1448/model_paras.pkl"
    # model_para_path_B = r"/data/hjj/ssh_prediction/work_dir/scs/parameter_3b_gc_0.7/pinn_0.7_batchsize4/SV+GC_20251231_1236/model_paras.pkl"

    modelA = RNN(args.model_config)
    modelB = RNN(args.model_config)
    model_para_path_A = r"/data/hjj/ssh_prediction/work_dir/scs/RNN_seed42/patchsize/best/PR_20260408_1913/model_paras.pkl"
    model_para_path_B = r"/data/hjj/ssh_prediction/work_dir/scs/RNN_seed42/patchsize/best/PR+GC_20260409_0711/model_paras.pkl"
    modelA.load_state_dict(
        torch.load(model_para_path_A, weights_only=True)
    )
    modelB.load_state_dict(
        torch.load(model_para_path_B, weights_only=True)
    )


    ModelA = Model(modelA, args)
    ModelB = Model(modelB, args)

    dataset = MvDataset(args, mode='test', norm=args.norm)

    pearson_tool = MaskPearsonCorr(~mask_land)

    analyzer = WindStressAnalyzer(args, dataset, pearson_tool, ModelA, ModelB)

    analyzer.run_full_analysis_time(
        save_dir="/data/hjj/ssh_prediction/work_dir/scs/pinnVSbase/pr",
        mask_land=mask_land
    )