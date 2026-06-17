import numpy as np
import matplotlib.pyplot as plt

# 数据
data = {
    "SV": {"Base":2.01,"MI":1.86,"GC":1.79,"MI+GC":1.71},
    "PR": {"Base":2.03,"MI":1.64,"GC":1.91,"MI+GC":1.61},
    "PF": {"Base":2.21,"MI":2.26,"GC":2.49,"MI+GC":2.76}
}

models = list(data.keys())
strategies = ["Base","MI","GC","MI+GC"]

# 清新配色
colors = {
    "Base":"#B8C1CC",
    "MI":"#6FA8DC",
    "GC":"#F6B26B",
    "MI+GC":"#93C47D"
}

x = np.arange(len(models))
width = 0.18

plt.figure(figsize=(8,4.8))

for i,s in enumerate(strategies):
    values = [data[m][s] for m in models]
    plt.bar(x+(i-1.5)*width, values, width,
            color=colors[s],
            edgecolor="black",
            linewidth=0.7,
            label=s)

# 纵轴范围
plt.ylim(1.5,2.8)

# 字体整体放大
plt.xticks(x, models, fontsize=18)
plt.yticks(fontsize=16)

plt.xlabel("Model", fontsize=20)
plt.ylabel("RMSE (cm)", fontsize=20)

plt.grid(axis="y", linestyle="--", alpha=0.4)

# legend加背景板
legend = plt.legend(
    # title="Strategy",
    fontsize=18,
    # title_fontsize=18,
    frameon=True
)

legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.show()