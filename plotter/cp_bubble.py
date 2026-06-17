import matplotlib.pyplot as plt
import numpy as np

# 1. 核心数据 (基于标准配置筛选: pinn=0, ssh, gradient clipping)
data = {
    'SV': [[0.27, 0.88, 2.12], [3.27, 3.44, 2.01], [20.96, 15.14, 1.88]],
    'PF': [[0.21, 1.02, 3.00], [3.19, 4.27, 2.21], [25.29, 12.83, 2.29]],
    'PR': [[0.37, 8.99, 1.98], [4.11, 25.26, 2.03], [23.08, 114.48, 1.96]]
}

colors = {'SV': '#1f77b4', 'PF': '#ff7f0e', 'PR': '#2ca02c'}


# 气泡大小计算函数：次方极大化差异
def get_size(rmse):
    return (1 / rmse) ** 4 * 6e4


# 2. 绘图配置
plt.figure(figsize=(14, 11), dpi=150)
plt.rcParams.update({'font.size': 20})

scatter_handles = []
for model_name, points in data.items():
    params = [p[0] for p in points]
    latency = [p[1] for p in points]
    rmse_list = np.array([p[2] for p in points])
    sizes = get_size(rmse_list)

    sc = plt.scatter(params, latency, s=sizes, c=colors[model_name],
                     label=model_name, alpha=0.6, edgecolors='black', linewidth=1.5, zorder=3)
    scatter_handles.append(sc)

    # 动态标注
    for i in range(len(points)):
        # 计算气泡半径对应的偏移量 (sqrt(s)/2 是半径)
        radius_offset = (sizes[i] ** 0.5) / 2.0

        # 默认位置：正上方
        text_x, text_y = 0, radius_offset + 10

        # 特殊处理左侧重合区域 (Params < 1 的 US 模型)
        if params[i] < 1:
            if model_name == 'PF':  # 最左侧，稍微往左移
                text_x, text_y = -15, radius_offset + 5
            elif model_name == 'PR':  # 稍微往右偏
                text_x, text_y = 15, radius_offset + 5
            elif model_name == 'SV':  # 居中稍高
                text_x, text_y = 0, radius_offset + 15

        plt.annotate(f"{rmse_list[i]:.2f}", (params[i], latency[i]),
                     fontsize=18, fontweight='bold',
                     xytext=(text_x, text_y),
                     textcoords='offset points',
                     ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

# 3. 样式布局
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.1, 120)
plt.ylim(0.4, 600)

plt.xlabel('Parameters (M)', fontsize=22, labelpad=15)
plt.ylabel('Latency (ms)', fontsize=22, labelpad=15)
plt.title('Model Efficiency & Accuracy (Size $\propto$ 1/RMSE$^4$)', fontsize=26, pad=35)

# 4. 图例处理
legend1 = plt.legend(handles=scatter_handles, labels=data.keys(), title="Models",
                     loc='upper left', labelspacing=1.8, borderpad=1.5,
                     handletextpad=1.0, fontsize=20, title_fontsize=21, frameon=True)
for handle in legend1.legendHandles:
    handle._sizes = [600]

ref_rmses = [1.9, 2.2, 3.0]
ref_labels = [f"RMSE: {r} cm" for r in ref_rmses]
ref_points = [plt.scatter([], [], s=get_size(r), c='gray', alpha=0.3, edgecolors='black') for r in ref_rmses]

legend2 = plt.legend(ref_points, ref_labels, title="RMSE Scale",
                     loc='lower right', labelspacing=2.5, borderpad=1.5,
                     handletextpad=2.0, fontsize=18, title_fontsize=19, frameon=True)

plt.gca().add_artist(legend1)
plt.grid(True, which="both", ls="--", alpha=0.4, zorder=1)
plt.tight_layout()
plt.show()