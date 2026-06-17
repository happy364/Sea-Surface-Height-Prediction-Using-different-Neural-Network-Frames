import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

# 提取数据
rmse_data = {
    'PF-Summer': [0.6, 0.7, 1.0, 1.3, 1.6, 2.0, 2.4, 2.7, 3.1, 3.3],
    'PF-Winter': [0.7, 0.8, 1.1, 1.4, 1.9, 2.3, 2.7, 3.1, 3.5, 3.8],
    'PR-Summer': [1.2, 1.2, 1.2, 1.4, 1.6, 1.8, 2.1, 2.4, 2.8, 3.1],
    'PR-Winter': [0.6, 0.7, 0.9, 1.1, 1.4, 1.8, 2.2, 2.6, 3.0, 3.3],
    'SV-Summer': [0.3, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.1],
    'SV-Winter': [0.3, 0.4, 0.7, 1.0, 1.5, 1.9, 2.4, 2.9, 3.3, 3.7]
}

acc_data = {
    'PF-Summer': [0.996, 0.995, 0.991, 0.983, 0.973, 0.958, 0.942, 0.926, 0.909, 0.894],
    'PF-Winter': [0.997, 0.996, 0.992, 0.986, 0.977, 0.965, 0.951, 0.938, 0.924, 0.910],
    'PR-Summer': [0.991, 0.991, 0.989, 0.985, 0.979, 0.970, 0.957, 0.942, 0.925, 0.907],
    'PR-Winter': [0.998, 0.997, 0.996, 0.993, 0.987, 0.979, 0.969, 0.957, 0.943, 0.929],
    'SV-Summer': [0.999, 0.999, 0.997, 0.992, 0.984, 0.973, 0.958, 0.941, 0.922, 0.904],
    'SV-Winter': [0.999, 0.999, 0.997, 0.993, 0.987, 0.978, 0.966, 0.953, 0.940, 0.926]
}

# 计算季节性差异（冬季 - 夏季）
def calculate_seasonal_diff(data_dict):
    seasonal_diff = {}
    for model in ['PF', 'PR', 'SV']:
        summer_key = f'{model}-Summer'
        winter_key = f'{model}-Winter'
        seasonal_diff[model] = np.array(data_dict[winter_key]) - np.array(data_dict[summer_key])
    return seasonal_diff

# 计算差异
rmse_seasonal_diff = calculate_seasonal_diff(rmse_data)
acc_seasonal_diff = calculate_seasonal_diff(acc_data)

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

lead_times = np.arange(1, 11)

# RMSE季节性差异图（左图）
for model in ['PF', 'PR', 'SV']:
    ax1.plot(lead_times, rmse_seasonal_diff[model], marker='o', label=model, linewidth=2.5, markersize=6)
ax1.set_xlabel('Lead Time (Day)')
ax1.set_ylabel('cm')
ax1.set_title('RMSE Seasonal Difference')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(lead_times)
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# ACC季节性差异图（右图）
for model in ['PF', 'PR', 'SV']:
    ax2.plot(lead_times, acc_seasonal_diff[model], marker='s', label=model, linewidth=2.5, markersize=6)
ax2.set_xlabel('Lead Time (Day)')
ax2.set_title('ACC Seasonal Difference')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(lead_times)
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# 调整布局
plt.tight_layout()
plt.show()