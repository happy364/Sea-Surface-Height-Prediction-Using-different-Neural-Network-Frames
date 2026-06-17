"""
空间绘图完整示例 - 包含所有三个任务

任务 1: 多模型 MAE/Pred 空间分布对比
    - 读取多个模型的 npz 数据
    - 对时间维度平均 -> [10, 160, 160]
    - 画出 lead=1,4,7,10 的空间分布图
    - 可选择包含 Target 作为参照

任务 2: 季节性 MAE 差异分析
    - 读取 winter-a, winter-b, summer-a, summer-b
    - 计算 winter-a - winter-b (同季节不同模型/配置的差异)
    - 计算 summer-a - summer-b
    - 对时间维度平均后画图
    - 两行：Winter-Diff 和 Summer-Diff

任务 3: 完善 SpatialDataLoader
    - 已集成到 spatial_data_loader.py 中
    - 支持 6D 数据自动处理
    - 支持 target 数据添加
    - 支持季节性差异计算
"""

import numpy as np
from plotter.spatial_data_loader import SpatialDataLoader
from plotter.spatial_plotter import PlotterMultiPanel
import os
import glob
import  re
def find_npz_files(root_folder):
    """
    查找指定母文件夹中所有 .npz 文件，返回字典：
    key: 文件名（不带后缀）
    value: 文件完整路径

    :param root_folder: 母文件夹路径（字符串）
    :return: dict, {filename_without_ext: full_path}
    """
    if not os.path.exists(root_folder):
        raise FileNotFoundError(f"文件夹不存在: {root_folder}")
    if not os.path.isdir(root_folder):
        raise NotADirectoryError(f"路径不是文件夹: {root_folder}")

    # 使用 glob 递归查找所有 .npz 文件（包括子文件夹）
    pattern = os.path.join(root_folder, '**', '*.npz')
    npz_files = glob.glob(pattern, recursive=True)

    # 构建字典
    result = {}
    for file_path in npz_files:
        # 获取文件名（不带路径和后缀）
        filename = os.path.splitext(os.path.basename(file_path))[0]
        result[filename] = os.path.abspath(file_path)

    return result


def find_matching_file(model_files, model_prefix, season):
    print(f"🔎 查找: {model_prefix} + {season}")
    # 匹配文件名中包含 model_prefix 和 season
    pattern = re.compile(rf'(?=.*{re.escape(model_prefix)})(?=.*{re.escape(season)})', re.IGNORECASE)
    for file_path in model_files:
        filename = os.path.basename(file_path)
        if pattern.search(filename) and file_path.endswith('.npz'):
            print(f"  ✅ 匹配: {filename}")
            return file_path
    print(f"  ❌ 未找到")
    return None



def example_multi_model_mae(dir,lon,lat, extent):
    """
    任务 1: 多模型 MAE 空间分布对比

    数据流程:
        1. 加载多个模型的 npz 文件 (shape: [512, 1, 10, 1, 160, 160])
        2. squeeze 多余维度 -> [512, 10, 160, 160]
        3. 对时间维度平均 -> [10, 160, 160]
        4. 提取 lead=1,4,7,10 (索引 0,3,6,9) -> 4 个 [160, 160]
        5. 使用 PlotterMultiPanel 画图 (模型=行，lead=列)
    """
    print("=" * 70)
    print("任务 1: 多模型 MAE 空间分布对比")
    print("=" * 70)

    # ========== 步骤 1: 准备数据 ==========
    print("\n[步骤 1] 创建 SpatialDataLoader")

    lead_steps = list(range(1, 11))   # [1,2,3,...,10]

    loader = SpatialDataLoader(lon=lon, lat=lat, lead_steps=lead_steps)

    # ========== 步骤 2: 加载模型数据 ==========
    print("\n[步骤 2] 加载模型数据")

    model_files = find_npz_files(dir)

    # 检查文件是否存在
    for name, path in model_files.items():
        if not os.path.exists(path):
            print(f"! 文件不存在：{path}")
            return False

    loader.load_models(model_files)

    # ========== 步骤 3: 准备数据用于绘图 ==========
    print("\n[步骤 3] 准备数据用于 plot_panel()")

    # 选择 lead=1,4,7,10 (索引为 0,3,6,9)
    leads_to_plot = [0, 3, 6, 9]
    lead_labels = ['Lead 1', 'Lead 4', 'Lead 7', 'Lead 10']

    # 准备 MAE 数据
    data = loader.prepare_data_for_panel(
        data_type='mae_spatial',
        leads=leads_to_plot,
        include_target=False  # MAE 不需要 target，如果画 pred 设置为 True
    )

    # 自动计算 colorbar 范围
    cbar_range = loader.get_cbar_range('mae_spatial', leads_to_plot, percentile=(10, 99))
    print(f"  Colorbar 范围：{cbar_range}")
    cbar_range = { 'data': (0.,0.1)}

    # ========== 步骤 4: 使用 PlotterMultiPanel 绘图 ==========
    print("\n[步骤 4] 绘制空间分布图")

    plotter = PlotterMultiPanel(lon, lat, extent)

    # 使用更好看的 colormap: 'viridis' 或 'plasma'
    # viridis: 黄->绿->蓝，科学友好，感知均匀
    # plasma: 紫->粉->黄，更鲜艳
    # 'YlOrRd': 黄->橙->红，温暖色调
    plotter.plot_panel(
        data=data,
        cbar_range=cbar_range,
        title='MAE',
        save_path=os.path.join(dir,'mae_multi_model.png'),
        cmap='Reds',
        lead_labels=lead_labels,
        row_label_fontsize=26,
        tick_labelsize=18,
        convert2cm=True,
    )


def example_multi_model_pred(dir,lon,lat, extent):
    """
    任务 1 变体：多模型 Pred 空间分布对比（包含 Target 参照）

    与 MAE 不同的地方:
        - data_type='preds_spatial'
        - include_target=True (添加 Target 行作为参照)
    """
    print("=" * 70)
    print("任务 1 变体：多模型 Pred 空间分布对比 (含 Target)")
    print("=" * 70)

    lead_steps = list(range(1, 11))

    loader = SpatialDataLoader(lon=lon, lat=lat, lead_steps=lead_steps)

    model_files = find_npz_files( dir)

    # 检查文件
    for name, path in model_files.items():
        if not os.path.exists(path):
            print(f"! 文件不存在：{path}")
            return False

    loader.load_models(model_files)

    leads_to_plot = [0, 3, 6, 9]

    # 准备 Pred 数据，包含 Target
    data = loader.prepare_data_for_panel(
        data_type='preds_spatial',
        leads=leads_to_plot,
        include_target=True  # 添加 Target 行
    )

    # 自动计算 colorbar 范围（使用 pred 数据）
    cbar_range = loader.get_cbar_range('preds_spatial', leads_to_plot,percentile=(0.01, 99.9))
    cbar_range = {'data': (0.88,1.62)} #todo

    plotter = PlotterMultiPanel(lon, lat, extent)

    plotter.plot_panel(
        data = data,
        cbar_range=cbar_range,
        title='SSH',
        save_path=os.path.join(dir,'pred_multi_model_with_target.png'),
        cmap= 'RdBu_r',# 'seismic', # 红蓝反转，经典高度场 colormap
        lead_labels=['Lead 1', 'Lead 4', 'Lead 7', 'Lead 10'],
        row_label_fontsize=22,
        convert2cm=True,
        tick_labelsize=18,
    )

    print(f"\nOK 完成！图像已保存为 pred_multi_model_with_target.png")
    return True


def example_seasonal_model_diff(dir, lon, lat, extent):
    """
    任务 2: 季节性模型间差异分析 (正确的理解！)

    **重要**: 比较的是两个模型在同一季节的表现

    数据流程:
        1. 读取 4 个文件:
           - winterA.npz: 模型 A 在冬季 (MAE shape: [T, 10, 160, 160])
           - summerA.npz: 模型 A 在夏季
           - winterB.npz: 模型 B 在冬季
           - summerB.npz: 模型 B 在夏季

        2. 计算差异 (保持时间维度 T):
           - Winter Diff = Winter-A - Winter-B  [T, 10, 160, 160]
           - Summer Diff = Summer-A - Summer-B  [T, 10, 160, 160]

        3. 对时间维度 T 平均:
           - Winter_Diff_Mean -> [10, 160, 160]
           - Summer_Diff_Mean -> [10, 160, 160]

        4. 画图：
           - 两行：Winter(A-B) 和 Summer(A-B)
           - 四列：Lead 1, 4, 7, 10

    结果解释:
        - 红色 (正值): Model-A MAE > Model-B MAE → **B 模型更好**
        - 蓝色 (负值): Model-A MAE < Model-B MAE → **A 模型更好**
        - 白色 (0 值): 两模型表现相同
    """
    print("=" * 70)
    print("任务 2: 季节性模型间差异分析 (Winter A-B, Summer A-B)")
    print("=" * 70)

    lead_steps = list(range(1, 11))

    # model_files = find_npz_files(dir)
    # ========== 步骤 1: 定义 4 个文件路径 ==========
    prefix = 'SV'
    # model_a_winter = find_matching_file(model_files, f"{prefix}", "Winter")
    # model_a_summer = find_matching_file(model_files, f"{prefix}", "Summer")
    # model_b_winter = find_matching_file(model_files, f"{prefix}+GC", "Winter")
    # model_b_summer = find_matching_file(model_files, f"{prefix}+GC", "Summer")

    # 1. 指定目录和文件路径
    # model_dir = "/data/hjj/ssh_prediction/work_dir/scs/season_diff_spatiol/diff"

    #
    model_a_winter = f"{dir}/{prefix}-Winter.npz"
    model_a_summer = f"{dir}/{prefix}-Summer.npz"
    model_b_winter = f"{dir}/{prefix}+GC-Winter.npz"
    model_b_summer = f"{dir}/{prefix}+GC-Summer.npz"



    print(f"\n颜色解释:")
    print(f"  - 红色 (正值): Model-A MAE > Model-B MAE → B 模型表现更好")
    print(f"  - 蓝色 (负值): Model-A MAE < Model-B MAE → A 模型表现更好")
    print(f"  - 白色 (0): 两模型表现相同")

    # ========== 步骤 2: 使用 load_seasonal_model_diff ==========
    print("\n[步骤 2] 加载数据并计算模型间差异")
    print("  计算逻辑:")
    print("    Winter Diff = Winter-A - Winter-B")
    print("    Summer Diff = Summer-A - Summer-B")

    loader = SpatialDataLoader.load_seasonal_model_diff(
        lon=lon, lat=lat, lead_steps=lead_steps,
        model_a_winter_path=model_a_winter,
        model_a_summer_path=model_a_summer,
        model_b_winter_path=model_b_winter,
        model_b_summer_path=model_b_summer,
        model_a_name=f'{prefix}',
        model_b_name=f'{prefix}+GC'
    )

    # ========== 步骤 3: 准备数据 ==========
    print("\n[步骤 3] 准备数据用于绘图")

    leads_to_plot = [0, 3, 6, 9]
    data = {}
    for model_name, model_data in loader.models.items():
        mae_diff = model_data['mae_spatial']  # shape=(10, 160, 160)
        data[model_name] = np.array([mae_diff[lead_idx] for lead_idx in leads_to_plot])
        print(f"  准备 {model_name}: shape={mae_diff.shape}")

    # 计算对称 colorbar range (关于 0 对称)
    # all_diff = np.concatenate([np.array(arr).ravel() for arr in data.values()])
    # all_diff = all_diff[~np.isnan(all_diff)]
    # abs_max = np.nanpercentile(np.abs(all_diff), 96)
    cbar_range = {'data': (-0.01, 0.01)}
    print(f"\n  Colorbar 范围：{cbar_range} (对称，0 为中心)")

    # ========== 步骤 4: 绘图 ==========
    print("\n[步骤 4] 绘制差异性图")

    plotter = PlotterMultiPanel(lon, lat, extent)

    plotter.plot_panel(
        data=data,
        cbar_range=cbar_range,
        title='Seasonal Mean MAE Difference',
        save_path=os.path.join(dir,'seasonal_model_diff.png'),
        cmap='RdBu_r',  # 红蓝反转，0 为白色
        lead_labels=['Lead 1', 'Lead 4', 'Lead 7', 'Lead 10'],
        row_label_fontsize=22,
        convert2cm=True,
        tick_labelsize=18,
    )

    print(f"\nOK 完成！图像已保存为 seasonal_model_diff.png")
    print(f"\n说明:")
    print(f"  - 第一行：Model-A - Model-B (冬季)")
    print(f"  - 第二行：Model-A - Model-B (夏季)")
    print(f"  - 每行 4 列：Lead 1, 4, 7, 10")
    print(f"\n颜色解释:")
    print(f"  - 红色 (正值): Model-A MAE > Model-B MAE → B 模型表现更好")
    print(f"  - 蓝色 (负值): Model-A MAE < Model-B MAE → A 模型表现更好")
    print(f"  - 白色 (0): 两模型表现相同")

    return True




if __name__ == "__main__":
    """
    使用说明:
    
    1. 根据你的数据路径修改 model_files, winter_files, summer_files
    2. npz 文件应包含 'mae_spatial', 'preds_spatial', 'targets_spatial' 等键
    3. 运行此脚本会自动:
       - 加载数据 (处理 6D -> 4D -> 3D)
       - 对时间维度平均
       - 提取指定的 lead
       - 使用 PlotterMultiPanel 绘图
    
    生成的图像:
       - mae_multi_model.png: 任务 1 - 多模型 MAE 对比
       - pred_multi_model_with_target.png: 任务 1 变体 - Pred 对比 (含 Target)
       - seasonal_diff.png: 任务 2 - 季节性差异分析
    """
    import numpy as np
    from configs import parse_args, get_my_config
    from dataset import MvDataset

    args_ = parse_args()
    args_.need_ssh = True
    # args_.env = "windows"
    # args_.base = r"/data/hjj/SEJ/data/analyze/indian_ocean_data"

    args = get_my_config(args_)
    test_dataset = MvDataset(args, mode='test', norm=True)
    lon, lat = test_dataset.lon, test_dataset.lat
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    # extent = [40, 100, -30, 30] #indian ocean
    print("lon, lat:", lon.shape, lat.shape)
    print(f"lon :({lon.min()} ~ {lon.max()}) ({lat.min()} ~ {lat.max()})")
    extent = [104, 124, 2, 22]
    #
    base_dir = r"/data/hjj/SEJ/analyze/model_paras/case-study（2-10起报）/spatial/input"
    base_dir = r"/data/hjj/SEJ/analyze/model_paras/case_study-2-20起报/phys"
    base_dir = r"/data/hjj/ssh_prediction/work_dir/scs/RNN_seed42/patchsize/full"
    # 任务 1: 多模型 MAE 对比
    print("\n\n>>> 运行任务 1: 多模型 MAE 对比\n")
    try:
        example_multi_model_mae(base_dir,lon,lat, extent)
    except Exception as e:
        print(f"! 任务 1 出错：{e}")
        import traceback
        traceback.print_exc()

    print("\n\n")
    #任务 1 变体: Pred 对比
    print(">>> 运行任务 1 变体：多模型 Pred 对比 (含 Target)\n")
    try:
        example_multi_model_pred(base_dir,lon,lat, extent)
    except Exception as e:
        print(f"! 任务 1 变体出错：{e}")
        import traceback
        traceback.print_exc()

    print("\n\n")

    # 任务 2: 季节性差异
    print(">>> 运行任务 2: 季节性 MAE 差异分析\n")
    season_dir = "/data/hjj/ssh_prediction/work_dir/scs/season_diff_spatiol/diff"
    # season_dir = r"/data/hjj/ssh_prediction/work_dir/scs/RNN_seed42/patchsize/season-diff"
    try:
        example_seasonal_model_diff(season_dir,lon,lat, extent)
    except Exception as e:
        print(f"! 任务 2 出错：{e}")
        import traceback
        traceback.print_exc()

    print("=" * 70)
    print("所有任务完成！请根据实际数据路径修改脚本中的文件路径配置")
    print("=" * 70)
