import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np


def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence_level=0.95):
    """
    使用bootstrap方法计算置信区间
    """
    if len(data) == 0:
        return 0, 0, 0

    if len(data) == 1:
        return data[0], data[0], data[0]

    # 计算均值
    sample_mean = np.mean(data)

    # bootstrap采样
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    # 计算置信区间
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return sample_mean, lower_bound, upper_bound


def plot_ablation_study(csv_path, n_exe=5, n_bootstrap=10000, confidence_level=0.95):
    """
    绘制消融实验结果的beeswarm图和点线图
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 定义要处理的模型类型及其显示名称，按指定顺序排列
    model_types = {
        'mask_pinn_0.700': {'display_name': 'Phys-SV', 'style': 'point'},
        'mask_pinn_0.700_no_weight': {'display_name': 'No LW', 'style': 'point'},
        'pinn_0.700': {'display_name': 'No MI', 'style': 'point'},
        'mask_pinn_0.000': {'display_name': 'No GC', 'style': 'point'},
        'base': {'display_name': 'Base-SV', 'style': 'point'}
    }

    # 存储所有模型的数据
    model_data = {}

    for model_key, model_info in model_types.items():
        row = df[df['key'] == model_key]
        if not row.empty:
            # 获取原始数据
            rmse_values = []
            corr_values = []

            # 提取RMSE值
            for i in range(1, 1 + n_exe):
                val = row.iloc[0, i]
                if pd.notna(val) and val != '':
                    rmse_values.append(float(val))

            # 提取Corr值
            corr_start_idx = 1 + n_exe + 2 + n_exe + 2
            for i in range(corr_start_idx, corr_start_idx + n_exe):
                val = row.iloc[0, i]
                if pd.notna(val) and val != '':
                    corr_values.append(float(val))

            # 计算bootstrap置信区间
            rmse_mean, rmse_lower, rmse_upper = bootstrap_confidence_interval(
                rmse_values, n_bootstrap, confidence_level) if rmse_values else (0, 0, 0)

            corr_mean, corr_lower, corr_upper = bootstrap_confidence_interval(
                corr_values, n_bootstrap, confidence_level) if corr_values else (0, 0, 0)

            model_data[model_key] = {
                'display_name': model_info['display_name'],
                'style': model_info['style'],
                'rmse_values': rmse_values,
                'rmse_mean': rmse_mean,
                'rmse_lower': rmse_lower,
                'rmse_upper': rmse_upper,
                'corr_values': corr_values,
                'corr_mean': corr_mean,
                'corr_lower': corr_lower,
                'corr_upper': corr_upper
            }

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 增加宽度以提供更多空间

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 定义颜色，按新顺序排列
    colors = {
        'mask_pinn_0.700': '#2ca02c',  # 绿色 - GCNN
        'mask_pinn_0.700_no_weight': '#ff7f0e',  # 橙色 - No LW
        'pinn_0.700': '#9467bd',  # 紫色 - No MI
        'mask_pinn_0.000': '#1f77b4',  # 蓝色 - No GC
        'base': '#d62728'  # 红色 - Base
    }

    markers = ['o', 's', '^', 'D', 'v']

    # 第一张子图：RMSE
    x_positions = range(len([m for m in model_data if model_data[m]['style'] == 'point']))
    point_model_counter = 0
    spacing_factor = 0.6  # 控制点水平间距的紧凑程度，越小越紧凑
    x_offset = 0  # 添加偏移量使数据点整体向右偏移，增加左边距

    # 绘制点线图和beeswarm图（所有模型）
    for i, (model_key, data) in enumerate(model_data.items()):
        if data['style'] == 'point' and data['rmse_values']:
            color = colors[model_key]
            marker = markers[i % len(markers)]

            # 绘制beeswarm图
            jitter = np.random.normal(0, 0.05, len(data['rmse_values']))
            ax1.scatter([i * spacing_factor + j + x_offset for j in jitter], data['rmse_values'],
                        alpha=0.6, color=color, s=50, zorder=2,
                        edgecolors='white', linewidth=0.5)

            # 绘制点线图（均值和置信区间）
            rmse_err_lower = data['rmse_mean'] - data['rmse_lower']
            rmse_err_upper = data['rmse_upper'] - data['rmse_mean']

            ax1.errorbar(i * spacing_factor + x_offset, data['rmse_mean'],
                         yerr=[[rmse_err_lower], [rmse_err_upper]],
                         marker=marker, markersize=10, linewidth=2,
                         capsize=8, capthick=2, elinewidth=2,
                         color=color, ecolor=color,
                         zorder=3)

    ax1.set_ylabel('RMSE (m)', fontsize=22)
    ax1.set_title('(a) RMSE Comparison', fontsize=24, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.set_xticks([i * spacing_factor + x_offset for i in range(len(model_data))])
    ax1.set_xticklabels([model_data[key]['display_name'] for key in model_data.keys()])
    ax1.set_xlim(-0.2, (len(model_data) - 1) * spacing_factor + x_offset + 0.2)

    # 第二张子图：PCC
    point_model_counter = 0

    # 绘制点线图和beeswarm图（所有模型）
    for i, (model_key, data) in enumerate(model_data.items()):
        if data['style'] == 'point' and data['corr_values']:
            color = colors[model_key]
            marker = markers[i % len(markers)]

            # 绘制beeswarm图
            jitter = np.random.normal(0, 0.05, len(data['corr_values']))
            ax2.scatter([i * spacing_factor + j + x_offset for j in jitter], data['corr_values'],
                        alpha=0.6, color=color, s=50, zorder=2,
                        edgecolors='white', linewidth=0.5)

            # 绘制点线图（均值和置信区间）
            corr_err_lower = data['corr_mean'] - data['corr_lower']
            corr_err_upper = data['corr_upper'] - data['corr_mean']

            ax2.errorbar(i * spacing_factor + x_offset, data['corr_mean'],
                         yerr=[[corr_err_lower], [corr_err_upper]],
                         marker=marker, markersize=10, linewidth=2,
                         capsize=8, capthick=2, elinewidth=2,
                         color=color, ecolor=color,
                         zorder=3)

    # 设置y轴刻度间隔更稀疏
    from matplotlib.ticker import MaxNLocator
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))  # 减少y轴刻度数量

    ax2.set_ylabel('PCC', fontsize=22)
    ax2.set_title('(b) PCC Comparison', fontsize=24, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.set_xticks([i * spacing_factor + x_offset for i in range(len(model_data))])
    ax2.set_xticklabels([model_data[key]['display_name'] for key in model_data.keys()])
    ax2.set_xlim(-0.2, (len(model_data) - 1) * spacing_factor + x_offset + 0.2)

    # 设置图表整体布局
    plt.tight_layout()  # 启用tight_layout来自动调整子图间距

    # 保存图表
    output_path = csv_path.parent / "ablation_study_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")

    # 打印统计结果
    print("\n=== 消融实验统计结果 ===")
    for model_key in model_types.keys():
        if model_key in model_data:
            data = model_data[model_key]
            print(f"\n{data['display_name']}:")
            print(f"  RMSE: {data['rmse_mean']:.3f} ({data['rmse_lower']:.3f}-{data['rmse_upper']:.3f})")
            print(f"  PCC: {data['corr_mean']:.3f}% ({data['corr_lower']:.3f}-{data['corr_upper']:.3f})")

    plt.close(fig)


def main(parent_directory=None, n_exe=5, n_bootstrap=10000):
    """
    主函数
    
    参数:
        parent_directory: 包含CSV文件的目录路径
        n_exe: 每个模型的实验次数
        n_bootstrap: Bootstrap抽样次数
        
    返回:
        fig: 生成的图表对象
    """
    # 如果未提供目录路径，则使用默认路径
    if parent_directory is None:
        parent_directory = Path("D:\Data\sej\model_paras\sigmoid weight")
    else:
        parent_directory = Path(parent_directory)
    
    csv_path = parent_directory / "model_statistics.csv"

    # 检查文件是否存在
    if not csv_path.exists():
        raise FileNotFoundError(f"文件 {csv_path} 不存在")
    
    # 记录运行信息
    print(f"开始绘制消融实验图表...")
    print(f"使用CSV文件: {csv_path}")
    print(f"实验次数: {n_exe}, Bootstrap抽样次数: {n_bootstrap}")
    
    # 绘制图表
    plot_ablation_study(csv_path, n_exe=n_exe, n_bootstrap=n_bootstrap)


if __name__ == "__main__":
    try:
        fig = main()
        plt.show()  # 显示生成的图表
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise