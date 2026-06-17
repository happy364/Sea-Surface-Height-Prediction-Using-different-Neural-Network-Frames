"""
改进的时间序列绘图模块 - 统一样式系统 V2
========================================
"""

import os
from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


# ==================== ModelStyle 样式配置类 ====================

@dataclass
class ModelStyle:
    """单个模型的样式配置"""
    color: str
    marker: str
    linestyle: str
    alpha: float = 1.0
    linewidth: float = 2.0
    markersize: float = 8.0


# ==================== MultiModelData 多模型数据类 ====================

@dataclass
class MultiModelData:
    """多模型数据统一封装类（唯一的数据输入格式）"""
    model_names: List[str]
    style_map: Dict[str, ModelStyle]
    data_dict: Dict[str, np.ndarray]
    extra_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        """验证数据一致性"""
        assert self.model_names, "模型列表不能为空"
        assert all(name in self.style_map for name in self.model_names), "所有模型必须有样式配置"

        for metric_name, data in self.data_dict.items():
            assert data.shape[0] == self.num_models, f"{metric_name} 的第一维必须等于模型数 ({self.num_models})"

    @classmethod
    def from_data(cls, model_names: List[str], **data_dict):
        """从数据快速创建 MultiModelData（自动分配样式）"""
        # 使用 itertools.cycle 优雅地循环分配样式
        colors = cycle(["tab:blue", "tab:red", "tab:green", "tab:gray", "tab:purple", "tab:brown", "tab:pink"])
        markers = cycle(["o", "D", "s", "^", "v", "p", "h", "*"])
        linestyles = cycle(["-", "--", "-.", ":"])

        style_map = {
            name: ModelStyle(color=next(colors), marker=next(markers), linestyle=next(linestyles))
            for name in model_names
        }

        return cls(model_names=model_names, style_map=style_map, data_dict=data_dict)

    def get_style(self, model_name: str) -> ModelStyle:
        return self.style_map[model_name]

    def has_data(self, metric_name: str) -> bool:
        return metric_name in self.data_dict

    @property
    def num_models(self) -> int:
        return len(self.model_names)


# ==================== 基类 ====================

class BasePlotter:
    """通用绘图基类"""
    def __init__(self, figsize=(16, 8), dpi=300):
        self.figsize, self.dpi = figsize, dpi

    def save_figure(self, fig, save_path, filename, close=True):
        """保存图像"""
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        if close: plt.close(fig)
        return filepath

    @staticmethod
    def style_axes(ax, show_xlabel=False, fontsize=24):
        """设置坐标轴样式"""
        ax.tick_params(axis="both", labelsize=fontsize)
        if not show_xlabel: ax.set_xlabel("")
        if ylabel := ax.get_ylabel(): ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=5, prune=None))


# ==================== LeadPlotter ====================

class LeadPlotter(BasePlotter):
    """预测步长指标绘图类"""

    def __init__(self, multi_model_data: MultiModelData, figsize=(16, 8), dpi=300):
        super().__init__(figsize, dpi)
        self.multi_model = multi_model_data
        self.model_names = multi_model_data.model_names
        self.num_models = multi_model_data.num_models

        if not multi_model_data.has_data("rmse_lead"):
            raise ValueError("MultiModelData 必须包含 'rmse_lead' 数据用于 LeadPlotter")
        self.num_steps = multi_model_data.data_dict["rmse_lead"].shape[1]

    def plot_metric(self, ax, data, metric, plot_type, title_label):
        """绘制指标图"""
        forecast_steps = np.arange(1, data.shape[1] + 1)
        bar_width = 0.8 / self.num_models

        for i, (name, row_data) in enumerate(zip(self.model_names, data)):
            style = self.multi_model.get_style(name)
            # 保持原有特定名称替换逻辑
            label = 'Phys-SV' if name == 'SV-Phys-None' else name.replace("-None", "")

            if plot_type == "line":
                ax.plot(forecast_steps, row_data, marker=style.marker, linestyle=style.linestyle,
                        linewidth=style.linewidth, markersize=style.markersize,
                        label=label, color=style.color, alpha=style.alpha)
            elif plot_type == "bar":
                x_positions = forecast_steps + (i - self.num_models / 2 + 0.5) * bar_width
                ax.bar(x_positions, row_data, width=bar_width,
                       color=style.color, label=label, alpha=style.alpha)

        # 提取公共坐标轴设置，减少重复代码
        ax.set_xticks(forecast_steps)
        ax.set_xticklabels([str(f) for f in forecast_steps], fontsize=24)
        ax.set_xlim(forecast_steps[0] - 0.5, forecast_steps[-1] + 0.5)
        ax.set_ylabel("RMSE (cm)" if metric == "rmse_lead" else "ACC", fontsize=24)
        ax.set_xlabel("Lead Time (Day)", fontsize=24)
        ax.set_title(title_label, loc="center", fontsize=22)
        ax.grid(True, linestyle="--", alpha=0.3)
        return ax

    def plot_two_panel(self, save_path, filename="two_panel.png"):
        """绘制双面板图：RMSE 线图 + ACC 线图"""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        if self.multi_model.has_data("rmse_lead"):
            self.plot_metric(axes[0], self.multi_model.data_dict["rmse_lead"], "rmse_lead", "line", "(a)")
        if self.multi_model.has_data("corr_lead"):
            self.plot_metric(axes[1], self.multi_model.data_dict["corr_lead"], "corr_lead", "line", "(b)")

        for j, ax in enumerate(axes):
            self.style_axes(ax, show_xlabel=True)
            if j == 0: ax.legend(fontsize=22)

        fig.tight_layout()
        return self.save_figure(fig, save_path, filename)

    def plot_heatmap(self, metric="rmse_lead", save_path=".", filename=None):
        """绘制热力图"""
        if not self.multi_model.has_data(metric):
            raise ValueError(f"指标 '{metric}' 不存在")

        data = self.multi_model.data_dict[metric]
        is_rmse = (metric == "rmse_lead")
        metric_label = "RMSE" if is_rmse else "ACC"
        unit_label = "RMSE (cm)" if is_rmse else "ACC"

        fig, ax = plt.subplots(figsize=(self.num_steps * 0.8 + 2, self.num_models * 0.6 + 1))

        sns.heatmap(data, ax=ax,
                    xticklabels=[str(i + 1) for i in range(self.num_steps)],
                    yticklabels=self.model_names,
                    cmap="RdYlGn_r" if is_rmse else "RdYlGn",
                    annot=True, fmt=".2f",
                    cbar_kws={"label": unit_label})

        ax.set_xlabel("Lead Time (Day)", fontsize=16)
        ax.set_title(f"{metric_label} Heatmap", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

        fig.tight_layout()
        return self.save_figure(fig, save_path, filename or f"{metric}_heatmap.png")


# ==================== FullYearPlotter ====================

class FullYearPlotter(BasePlotter):
    """全年时间序列绘图类"""

    def __init__(self, dates, multi_model: MultiModelData, figsize=(16, 8), dpi=300):
        super().__init__(figsize, dpi)
        self.dates = pd.to_datetime(dates)
        self.multi_model = multi_model
        self.num_models = multi_model.num_models
        self.num_days = len(dates)

        if not multi_model.has_data("rmse_date"):
            raise ValueError("MultiModelData 必须包含 'rmse_date' 数据用于 FullYearPlotter")

    def _add_peak_annotation(self, ax, df_filtered, model_name):
        """在图中找到指定模型的峰值并绘制虚线与标注"""
        col_name = f"{model_name}_value"
        if col_name not in df_filtered.columns:
            return

        peak_idx = df_filtered[col_name].idxmax()
        peak_row = df_filtered.loc[peak_idx]
        peak_date, peak_value = peak_row["Date"], peak_row[col_name]

        ax.axvline(x=peak_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

        ax.annotate(
            f"{peak_date.strftime('%Y-%m-%d')}\n{peak_value:.3f}",
            xy=(peak_date, peak_value),
            xytext=(8, 10), textcoords='offset points',
            fontsize=16, color='black', fontweight='bold',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.6)
        )

    def plot_time_series(self, ax=None, metric_name="RMSE", metric_unit="cm",
                         start_date=None, end_date=None, title=None, ylim=None,
                         highlight_peak_model='rest'):
        """绘制时间序列图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        val_key = "rmse_date" if metric_name == "RMSE" else "corr_date"

        # 向量化构建 DataFrame，避免 for 循环逐个插入列
        df = pd.DataFrame(
            self.multi_model.data_dict[val_key].T,
            columns=[f"{name}_value" for name in self.multi_model.model_names],
            index=self.dates
        )
        df["Date"] = self.dates

        if start_date: df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date: df = df[df["Date"] <= pd.to_datetime(end_date)]

        for name in self.multi_model.model_names:
            style = self.multi_model.get_style(name)
            ax.plot(df["Date"], df[f"{name}_value"], label=name,
                    color=style.color, linestyle=style.linestyle,
                    alpha=style.alpha, linewidth=min(style.linewidth, 1.5))

        if highlight_peak_model:
            self._add_peak_annotation(ax, df, highlight_peak_model)

        self._set_axes_style(ax, metric_name, metric_unit, start_date, end_date, title, ylim)
        return ax

    def _set_axes_style(self, ax, metric_name, metric_unit, start_date, end_date, title, ylim):
        """设置坐标轴样式，杜绝使用 plt 状态机"""
        ax.set_ylabel(f"{metric_name} ({metric_unit})" if metric_unit else metric_name, fontsize=24)
        ax.set_xlabel("Date", fontsize=24)
        if ylim: ax.set_ylim(ylim)
        if start_date and end_date:
            ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(fontsize=22, loc="upper right" if metric_name == "RMSE" else "lower right")
        ax.grid(True, linestyle="--", alpha=0.3)
        if title: ax.set_title(title, fontsize=22)
        ax.yaxis.set_major_locator(MaxNLocator(min_n_ticks=5, prune="upper", nbins=4))

        # 使用 ax 的方法替代 plt.xticks 和 plt.margins
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.margins(x=0)

    def save_plot(self, fig, save_path, metric_name="metric", start_date=None, end_date=None, filename=None):
        """保存图像 (修复了原代码中未定义的 self.metric_name)"""
        if filename is None:
            date_str = f"{start_date}_{end_date}" if start_date and end_date else "full"
            filename = f"{metric_name.lower()}_{date_str}.png"
        return self.save_figure(fig, save_path, filename)