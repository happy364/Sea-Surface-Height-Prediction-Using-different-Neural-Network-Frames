import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np

def extract_pinn_value(key):
    """
    从key中提取pinn值，只精确匹配mask_pinn_后跟数字的模式
    """
    # 使用 ^ 和 $ 确保完全匹配，只匹配形如 mask_pinn_数字 的字符串
    match = re.search(r'^mask_pinn_(\d+\.?\d*)$', key)
    if match:
        return float(match.group(1))
    return None


def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence_level=0.95):
    """
    使用bootstrap方法计算置信区间
    
    Parameters:
    data: 数据列表
    n_bootstrap: bootstrap采样次数
    confidence_level: 置信水平
    
    Returns:
    mean: 均值
    lower: 置信区间下界
    upper: 置信区间上界
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
        # 有放回采样
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # 计算置信区间
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return sample_mean, lower_bound, upper_bound

def plot_rmse_corr_beeswarm_with_bootstrap(csv_path, n_exe=5, n_bootstrap=10000, confidence_level=0.95):
    """
    从CSV文件中读取数据并绘制RMSE和Corr的beeswarm图叠加bootstrap点线图
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 提取base值（如果存在）
    base_row = df[df['key'] == 'base']
    base_rmse_mean = None
    base_rmse_lower = None
    base_rmse_upper = None
    base_corr_mean = None
    base_corr_lower = None
    base_corr_upper = None

    if not base_row.empty:
        # 获取base的原始数据进行bootstrap计算
        base_rmse_values = []
        base_corr_values = []
        
        # 从原始数据列中提取RMSE值（列1到n_exe）
        for i in range(1, 1 + n_exe):
            val = base_row.iloc[0, i]
            if pd.notna(val) and val != '':
                base_rmse_values.append(val)  # 转换回原始值
                

        corr_start_idx = 1 + n_exe + 2 + n_exe + 2
        for i in range(corr_start_idx, corr_start_idx + n_exe):
            val = base_row.iloc[0, i]
            if pd.notna(val) and val != '':
                base_corr_values.append(val)  # 转换回原始值
        
        # 计算base的bootstrap置信区间
        if base_rmse_values:
            base_rmse_mean, base_rmse_lower, base_rmse_upper = bootstrap_confidence_interval(
                base_rmse_values, n_bootstrap=n_bootstrap, confidence_level=confidence_level)
            
        if base_corr_values:
            base_corr_mean, base_corr_lower, base_corr_upper = bootstrap_confidence_interval(
                base_corr_values, n_bootstrap=n_bootstrap, confidence_level=confidence_level)

    # 筛选出key为类似mask_pinn_10.000的行
    mask_pinn_data = []
    
    for _, row in df.iterrows():
        pinn_value = extract_pinn_value(row['key'])
        if pinn_value is not None:
            # 获取该key的原始数据进行bootstrap计算
            rmse_values = []
            corr_values = []
            
            # 从原始数据列中提取RMSE值（列1到n_exe）
            for i in range(1, 1 + n_exe):
                val = row.iloc[i]
                if pd.notna(val) and val != '':
                    rmse_values.append(val)  
                    
            # 从原始数据列中提取Corr值
            corr_start_idx = 1 + n_exe + 2 + n_exe + 2
            for i in range(corr_start_idx, corr_start_idx + n_exe):
                val = row.iloc[i]
                if pd.notna(val) and val != '':
                    corr_values.append(val)  
            
            # 计算bootstrap置信区间
            if rmse_values:
                rmse_mean, rmse_lower, rmse_upper = bootstrap_confidence_interval(
                    rmse_values, n_bootstrap=n_bootstrap, confidence_level=confidence_level)
            else:
                rmse_mean, rmse_lower, rmse_upper = 0, 0, 0
                
            if corr_values:
                corr_mean, corr_lower, corr_upper = bootstrap_confidence_interval(
                    corr_values, n_bootstrap=n_bootstrap, confidence_level=confidence_level)
            else:
                corr_mean, corr_lower, corr_upper = 0, 0, 0
            
            mask_pinn_data.append({
                'key': row['key'],
                'pinn_value': pinn_value,
                'rmse_values': rmse_values,
                'rmse_mean': rmse_mean,
                'rmse_lower': rmse_lower,
                'rmse_upper': rmse_upper,
                'corr_values': corr_values,
                'corr_mean': corr_mean,
                'corr_lower': corr_lower,
                'corr_upper': corr_upper
            })

    # 转换为DataFrame并按pinn值排序
    if not mask_pinn_data:
        print("未找到符合要求的数据 (mask_pinn_*)")
        return

    plot_df = pd.DataFrame(mask_pinn_data)
    plot_df = plot_df.sort_values('pinn_value')

    # 创建等间距的x轴位置
    x_positions = range(len(plot_df))
    pinn_labels = plot_df['pinn_value'].tolist()

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 第一张子图：RMSE beeswarm图叠加bootstrap点线图
    # 绘制beeswarm图（使用原始数据点）
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['rmse_values']:
            # 添加一些随机抖动以创建beeswarm效果
            jitter = np.random.normal(0, 0.08, len(row['rmse_values']))
            ax1.scatter([i + j for j in jitter], np.array(row['rmse_values']) , 
                       alpha=0.7, color='#1f77b4', s=40, zorder=2, edgecolors='white', linewidth=0.5)

    # 绘制bootstrap均值点和置信区间线
    rmse_means = plot_df['rmse_mean'] 
    rmse_err_lower = (plot_df['rmse_mean'] - plot_df['rmse_lower']) 
    rmse_err_upper = (plot_df['rmse_upper'] - plot_df['rmse_mean']) 
    rmse_errors = [rmse_err_lower, rmse_err_upper]
    
    ax1.errorbar(x_positions, rmse_means,
                 yerr=rmse_errors, marker='o', linewidth=2,
                 markersize=8, capsize=5, capthick=2, elinewidth=2,
                 color='darkblue', ecolor='lightblue', label='Phys-SV', zorder=3)

    # 如果存在base值，则添加水平虚线
    if base_rmse_mean is not None:
        base_rmse_mean_mm = base_rmse_mean 
        base_rmse_lower_mm = base_rmse_lower 
        base_rmse_upper_mm = base_rmse_upper 
        
        ax1.axhline(y=base_rmse_mean_mm, color='#d62728', linestyle='--', linewidth=2,
                    label=f'Base-SV', zorder=1)
        # 绘制base模型的置信区
        ax1.axhspan(base_rmse_lower_mm, base_rmse_upper_mm,
                    color='#d62728', alpha=0.1, zorder=1)

    ax1.set_xlabel('$\lambda$', fontsize=20)
    ax1.set_ylabel('RMSE (m)', fontsize=20)
    ax1.set_title('(a) RMSE Comparison', fontsize=24, fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{val:.1f}' for val in pinn_labels])
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.legend(fontsize=20,loc='upper left')

    # 第二张子图：Corr beeswarm图叠加bootstrap点线图
    # 绘制beeswarm图（使用原始数据点）
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['corr_values']:
            # 添加一些随机抖动以创建beeswarm效果
            jitter = np.random.normal(0, 0.08, len(row['corr_values']))
            ax2.scatter([i + j for j in jitter], row['corr_values'], 
                       alpha=0.7, color='#ff7f0e', s=40, zorder=2, edgecolors='white', linewidth=0.5)

    # 绘制bootstrap均值点和置信区间线
    corr_means = plot_df['corr_mean']
    corr_err_lower = (plot_df['corr_mean'] - plot_df['corr_lower'])
    corr_err_upper = (plot_df['corr_upper'] - plot_df['corr_mean'])
    corr_errors = [corr_err_lower, corr_err_upper]
    
    ax2.errorbar(x_positions, corr_means,
                 yerr=corr_errors, marker='s', linewidth=2,
                 markersize=8, capsize=5, capthick=2, elinewidth=2,
                 color='darkorange', ecolor='moccasin', label='Phys-SV', zorder=3)

    # 如果存在base值，则添加水平虚线
    if base_corr_mean is not None:
        base_corr_mean_pct = base_corr_mean
        base_corr_lower_pct = base_corr_lower
        base_corr_upper_pct = base_corr_upper
        
        ax2.axhline(y=base_corr_mean_pct, color='#d62728', linestyle='--', linewidth=2,
                    label=f'Base-SV', zorder=1)
        # 绘制base模型的置信区间
        ax2.axhspan(base_corr_lower_pct, base_corr_upper_pct,
                    color='#d62728', alpha=0.1, zorder=1)

    ax2.set_xlabel('$\lambda$', fontsize=20)
    ax2.set_ylabel('PCC', fontsize=20)
    ax2.set_title('(b) PCC Comparison', fontsize=24, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'{val:.1f}' for val in pinn_labels])
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.legend(fontsize=20,loc='upper left')

    # 设置图表整体布局
    plt.tight_layout()

    # 保存图表
    output_path = csv_path.parent / "rmse_corr_beeswarm_bootstrap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")

    return fig, (ax1, ax2)

def main():
    """
    主函数
    """
    # 设置路径
    parent_directory = Path("D:\Data\sej\model_paras\sigmoid weight")
    csv_path = parent_directory / "model_statistics.csv"

    # 检查文件是否存在
    if not csv_path.exists():
        print(f"错误: 文件 {csv_path} 不存在")
        return

    # 绘制图表
    plot_rmse_corr_beeswarm_with_bootstrap(csv_path, n_exe=5, n_bootstrap=10000)


if __name__ == "__main__":
    main()
