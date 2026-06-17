import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime


# ==========================================
# 辅助函数：绘制季节背景阴影
# ==========================================
def add_season_backgrounds(ax, date_list):
    """根据日期列表绘制冬夏季节背景阴影"""
    start_date, end_date = pd.to_datetime(date_list[0]), pd.to_datetime(date_list[-1])

    for year in range(start_date.year - 1, end_date.year + 1):
        # 冬季：10.1 - 次年 3.31 (浅蓝)
        w_s, w_e = datetime(year, 10, 1), datetime(year + 1, 3, 31)
        if w_s <= end_date and w_e >= start_date:
            ax.axvspan(mdates.date2num(max(w_s, start_date)),
                       mdates.date2num(min(w_e, end_date)),
                       facecolor='lightblue', alpha=0.2, label='Winter (10-03)')

        # 夏季：4.1 - 9.30 (浅黄)
        s_s, s_e = datetime(year, 4, 1), datetime(year, 9, 30)
        if s_s <= end_date and s_e >= start_date:
            ax.axvspan(mdates.date2num(max(s_s, start_date)),
                       mdates.date2num(min(s_e, end_date)),
                       facecolor='wheat', alpha=0.2, label='Summer (04-09)')


# ==========================================
# 功能1：支持季节过滤的箱线图
# ==========================================
def plot_corr_box_seasonal(corr_matrix, p_matrix, dates, save_path, season_mode='all'):
    """
    dates: 对应样本的起始日期数组 (np.datetime64)
    season_mode: 'all', 'summer', 'winter'
    """
    N, T = corr_matrix.shape
    dt_objects = pd.to_datetime(dates)
    lead_times = [f"T+{t + 1}" for t in range(T)]

    data_list = []
    for t in range(T):
        for n in range(N):
            month = dt_objects[n].month
            is_summer = 4 <= month <= 9

            if (season_mode == 'summer' and not is_summer) or \
                    (season_mode == 'winter' and is_summer):
                continue

            data_list.append({
                'Lead Time': lead_times[t],
                'Correlation': corr_matrix[n, t],
                'Significant': p_matrix[n, t] < 0.05
            })

    df = pd.DataFrame(data_list)
    sig_counts = df.groupby('Lead Time')['Significant'].mean() * 100
    sig_counts = sig_counts.reindex(lead_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    sns.boxplot(data=df, x='Lead Time', y='Correlation', ax=ax1, palette="Blues", fliersize=1)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title(f"Correlation Distribution - {season_mode.upper()}")

    sns.barplot(x=sig_counts.index, y=sig_counts.values, ax=ax2, palette="Greens")
    ax2.set_ylabel("Sig. Ratio (%)")
    ax2.set_ylim(0, 100)

    for i, val in enumerate(sig_counts.values):
        ax2.text(i, val + 1, f"{val:.1f}%", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==========================================
# 功能2：带季节背景的显著性折线图
# ==========================================
def plot_corr_lines_seasonal(corr_matrix, p_matrix, dates, save_path):
    """仅绘制 T+1 和 T+10 的显著相关性，背景区分冬夏"""
    N, T = corr_matrix.shape
    dt_objects = pd.to_datetime(dates)

    fig, ax = plt.subplots(figsize=(15, 6))
    add_season_backgrounds(ax, dt_objects)

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, 2))  # 取两个明显的颜色
    target_steps = [T - 1]  # 对应 T+1 和 T+10

    for idx, t in enumerate(target_steps):
        y = corr_matrix[:, t].copy()
        y[p_matrix[:, t] >= 0.05] = np.nan  # 掩盖不显著点

        ax.plot(dt_objects, y, color=colors[idx], label=f'Lead Time T+{t + 1}',
                linewidth=1.2, alpha=0.9)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.axhline(0, color='black', alpha=0.3)

    # 图例去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    plt.title("Significant Correlations (p < 0.05) with Seasonal Backgrounds")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import pearsonr

    # === 读取 ===
    res = np.load("/data/hjj/ssh_prediction/work_dir/scs/pinnVSbase/curl_corr.npz")
    base_path = "/data/hjj/ssh_prediction/work_dir/scs/pinnVSbase"
    dates = res['dates']

    # ==========================================
    # 1️⃣ 自动绘制所有 corr / p（box + line）
    # ==========================================
    corr_keys = [k for k in res.keys() if k.startswith("corr_")]

    for corr_key in corr_keys:
        p_key = corr_key.replace("corr_", "p_")
        if p_key not in res:
            continue

        corr = res[corr_key]
        p = res[p_key]

        name = corr_key.replace("corr_", "")

        # --- box（全季 / 夏 / 冬）---
        for mode in ['all']:
            plot_corr_box_seasonal(
                corr, p, dates,
                f"{base_path}/box_{name}_{mode}.png",
                season_mode=mode
            )

        # --- line ---
        plot_corr_lines_seasonal(
            corr, p, dates,
            f"{base_path}/lines_{name}.png"
        )

    # ==========================================
    # 2️⃣ wind vs corr（自动匹配）
    # ==========================================
    if 'wind_mag_mean' in res:
        wind_mag_mean = res['wind_mag_mean']
        wind_change_mag_mean = res['wind_change_mag_mean']

        for corr_key in corr_keys:
            corr = res[corr_key]
            name = corr_key.replace("corr_", "")

            T = corr.shape[1]
            corr_list = np.zeros(T)
            corr_list_change = np.zeros(T)
            p_list = np.zeros(T)
            p_list_change = np.zeros(T)

            for t in range(T):
                x = wind_mag_mean[:, t]
                xx = wind_change_mag_mean[:, t]
                y = corr[:, t]
                corr_list[t], p_list[t] = pearsonr(x, y)
                corr_list_change[t], p_list_change[t] = pearsonr(xx, y)

            # === plot ===
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

            x_axis = np.arange(T)
            line, = ax.plot(x_axis, corr_list, label=name)
            color = line.get_color()

            line_change, = ax.plot(x_axis, corr_list_change, label=f"{name}_change")
            color_change = line_change.get_color()


            sig = p_list < 0.05
            sig_change = p_list_change < 0.05

            # 显著：实心
            ax.scatter(x_axis[sig], corr_list[sig],
                       s=30, color=color, zorder=3)
            ax.scatter(x_axis[sig_change], corr_list_change[sig_change],
                       s=30, color=color_change, zorder=3)

            # 不显著：空心
            ax.scatter(x_axis[~sig], corr_list[~sig],
                       s=30, facecolors='none', edgecolors=color, zorder=3)
            ax.scatter(x_axis[~sig_change], corr_list_change[~sig_change],
                       s=30, facecolors='none', edgecolors=color_change, zorder=3)

            ax.axhline(0, linestyle='--')
            ax.set_xlabel("Lead Time")
            ax.set_ylabel("Correlation")
            ax.set_title(f"WindMag vs {name}")

            ax.grid()
            ax.legend()

            plt.savefig(f"{base_path}/windmag_vs_{name}.png", dpi=300)
            plt.close()


