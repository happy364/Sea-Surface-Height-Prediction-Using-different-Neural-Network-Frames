import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotter.temporal_plotter import (
    ModelStyle, MultiModelData,
    LeadPlotter, FullYearPlotter
)

"""
现在 analyze.py 生成的结构为：

data_dir/
    model1.npz
    model2.npz
    Persistence.npz

每个文件内部包含：
    rmse_lead, corr_lead, rmse_date, corr_date
"""


# =========================================================
# 1. 加载所有模型文件
# =========================================================

def load_analyze_results(data_dir):
    data_path = Path(data_dir)
    model_files = sorted(data_path.glob("*.npz"))

    if not model_files:
        raise ValueError("目录中没有找到任何模型文件")

    print(f"发现模型文件：{[f.name for f in model_files]}")

    raw_data = {}
    for file_path in model_files:
        model_name = file_path.stem
        data = np.load(file_path)

        if "rmse_lead" not in data:
            print(f"⚠ {model_name} 不包含 lead 指标，跳过")
            continue

        # 统一读取并转换单位
        raw_data[model_name] = {
            "rmse_lead": data["rmse_lead"] * 100,
            "corr_lead": data["corr_lead"],
            "rmse_date": data["rmse_date"] * 100,
            "corr_date": data["corr_date"]
        }
        print(f"✓ 已加载 {model_name}")

    # === 排序模型：按 rmse_lead 的均值升序（性能从好到差） ===
    sorted_names = sorted(raw_data.keys(), key=lambda x: np.mean(raw_data[x]["rmse_lead"]))
    # sorted_names = ['SV-Summer','SV-Winter','PR-Summer','PR-Winter','PF-Summer','PF-Winter']

    # === 一次性堆叠所有数据为 numpy 数组 (shape: n_models, ...) ===
    metrics = ["rmse_lead", "corr_lead", "rmse_date", "corr_date"]
    stacked_data = {
        metric: np.stack([raw_data[name][metric] for name in sorted_names])
        for metric in metrics
    }

    # === 构造日期 ===
    n_days = stacked_data["rmse_date"].shape[1]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

    print(f"\n最终参与绘图的模型({len(sorted_names)}个)：{sorted_names}")

    return sorted_names, stacked_data, dates


# =========================================================
# 2. 创建 MultiModelData
# =========================================================

def create_unified_data(model_names, stacked_data, style_map=None):
    if style_map is None:
        print("   样式：自动分配")
        return MultiModelData.from_data(model_names=model_names, **stacked_data)

    print("   样式：自定义")
    return MultiModelData(
        model_names=model_names,
        style_map=style_map,
        data_dict=stacked_data
    )


# =========================================================
# 3. 主函数
# =========================================================

def plot_main(data_dir, save_dir, style_map=None):
    print("=" * 70)
    print("绘图主函数 - 新结构版本")
    print("=" * 70)

    # Step 1 & 2: 加载与构建
    print("\n[步骤 1 & 2] 加载模型文件并构建 MultiModelData...")
    model_names, stacked_data, dates = load_analyze_results(data_dir)
    multi_model = create_unified_data(model_names, stacked_data, style_map)

    # Step 3: 创建目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[步骤 3] 保存目录: {save_dir}")

    # Step 4: 绘制 Lead 图
    print("\n[步骤 4] 绘制 Lead 图...")
    lead_plotter = LeadPlotter(multi_model, figsize=(16, 8))

    results = {}
    results["lead_two_panel"] = lead_plotter.plot_two_panel(save_dir, "lead_two_panel.png")
    print(f"   ✓ 双面板图: {results['lead_two_panel']}")

    results["lead_rmse_heatmap"] = lead_plotter.plot_heatmap("rmse_lead", save_dir, "lead_rmse_heatmap.png")
    print(f"   ✓ RMSE 热力图: {results['lead_rmse_heatmap']}")

    if multi_model.has_data("corr_lead"):
        results["lead_corr_heatmap"] = lead_plotter.plot_heatmap("corr_lead", save_dir, "lead_corr_heatmap.png")
        print(f"   ✓ Corr 热力图: {results['lead_corr_heatmap']}")

    # Step 5: 时间序列
    print("\n[步骤 5] 绘制全年时间序列图...")
    full_plotter = FullYearPlotter(dates=dates, multi_model=multi_model)

    date_str_start, date_str_end = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")

    # 利用循环减少重复的画图代码
    time_series_configs = [
        ("RMSE", "cm", "Daily RMSE Time Series", "rmse_timeseries_full"),
        ("ACC", "", "Daily ACC Time Series", "corr_timeseries_full")
    ]

    for metric_name, unit, title, file_prefix in time_series_configs:
        fig, ax = plt.subplots(figsize=(16, 8))
        full_plotter.plot_time_series(ax, metric_name=metric_name, metric_unit=unit, title=title)

        filepath = full_plotter.save_plot(
            fig, save_dir, date_str_start, date_str_end, f"{file_prefix}.png"
        )
        results[file_prefix] = filepath
        print(f"   ✓ {metric_name} 时间序列: {filepath}")

    return results


# =========================================================
# 使用示例
# =========================================================

if __name__ == "__main__":
    data_dir = r"/data/hjj/ssh_prediction/work_dir/scs/parameter_3b_gc_0/pinn0_b4/full"
    save_dir = os.path.join(data_dir, "FIGURES")

    style_map = {
        'SV': ModelStyle(color="tab:blue", marker="o", linestyle="-"),
        'SV+GC': ModelStyle(color="tab:green", marker="D", linestyle="--"),
        'SV+MI': ModelStyle(color="tab:orange", marker="*", linestyle="-."),
        'Phys-SV': ModelStyle(color="tab:red", marker="x", linestyle="-"),
        'PR': ModelStyle(color="tab:orange", marker="D", linestyle="--"),
        'Persistence': ModelStyle(color="tab:gray", marker="^", linestyle=":"),
        'PF': ModelStyle(color="tab:green", marker="s", linestyle="-."),
        'ReST': ModelStyle(color="tab:red", marker="x", linestyle="-"),
        'STED': ModelStyle(color="tab:purple", marker="p", linestyle="-"),
    }
    # style_map =None

    plot_main(data_dir=data_dir, save_dir=save_dir, style_map=style_map)