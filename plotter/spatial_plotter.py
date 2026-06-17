import xarray as xr
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.ticker as mticker
import string

class Plotter:
    def __init__(self, lon, lat, extent, depth_path=None):
        self.lon, self.lat = lon, lat
        if lon.ndim == 1:
            self.lon, self.lat = np.meshgrid(lon, lat)

        self.extent = extent
        self.depth = None
        if depth_path:
            self.depth = xr.open_dataset(depth_path).depth.values

        plt.rcParams.update({
            'font.size': 18,
            'axes.labelsize': 20,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
        })

    # ---------- axis helpers ----------
    def init_ax(self, fig, spec):
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.set_extent(self.extent)
        self.add_geographical_features(ax)
        return ax

    def add_geographical_features(self, ax):
        ax.coastlines('10m', zorder=4)
        ax.add_feature(cfeature.LAND, edgecolor='black', zorder=2)

    def add_bathymetry(self, ax, levels=(-200,), **kwargs):
        if not hasattr(self, 'depth'):
            print("No bathymetry data")
            return
        ax.contour(
            self.lon, self.lat, self.depth,
            levels=levels,
            colors=kwargs.get('colors', 'black'),
            linewidths=kwargs.get('linewidths', 1.5),
            linestyles=kwargs.get('linestyles', '-'),
            transform=ccrs.PlateCarree()
        )

    # def add_gridlines(self, ax, left=True, bottom=True):
    #     gl = ax.gridlines(draw_labels=True, linestyle=':', linewidth=0.5)
    #     gl.top_labels = gl.right_labels = False
    #     gl.left_labels = left
    #     gl.bottom_labels = bottom

    def add_gridlines(self, ax, left=True, bottom=True):
        # 设置均匀的网格线间隔
        xticks = np.arange(np.floor(self.extent[0])-2,
                           np.ceil(self.extent[1]) , 4)
        yticks = np.arange(np.floor(self.extent[2])-2,
                           np.ceil(self.extent[3]) , 4)

        gl = ax.gridlines(
            draw_labels=True,
            linestyle=':',
            linewidth=0.5,
            xlocs=mticker.FixedLocator(xticks),  # 使用 FixedLocator
            ylocs=mticker.FixedLocator(yticks)  # 使用 FixedLocator
        )
        gl.top_labels = gl.right_labels = False
        gl.left_labels = left
        gl.bottom_labels = bottom

    # ---------- plot primitives ----------
    def plot_pcolor(self, ax, data, cmap, vmin, vmax):
        return ax.pcolormesh(
            self.lon, self.lat, data,
            shading='auto',
            cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree()
        )

    def plot_quiver(self, ax, u, v, step=5, scale=20):
        return ax.quiver(
            self.lon[::step, ::step],
            self.lat[::step, ::step],
            u[::step, ::step],
            v[::step, ::step],
            scale=scale,
            transform=ccrs.PlateCarree()
        )

    def add_colorbar(self, fig, pcm, pos, label=''):
        cax = fig.add_axes(pos)
        cbar = fig.colorbar(pcm, cax=cax)
        cbar.set_label(label)


class PlotterMultiPanel(Plotter):

    def plot_panel(self, data, cbar_range, title, save_path,
                   significance_data=None, hatch='////', # 👈 新增这两个参数
                   cmap='RdBu_r',
                   lead_labels=('Lead 1', 'Lead 4', 'Lead 7', 'Lead 10'),
                   row_label_fontsize=24, tick_labelsize=18,
                   convert2cm=False,):
        """
        绘制多面板空间分布图

        参数:
            data: {模型名：[lead1, lead2, lead3, lead4]}
            significance_data: {模型名：[mask1, mask2, mask3, mask4]}，布尔矩阵
            hatch: 斜线样式，如 '////', '...', '\\\\'
            ... (其余参数保持不变)
        """
        n_rows = len(data)
        n_cols = 4

        if convert2cm:
            for i, (row_name, row_data) in enumerate(data.items()):
                data[row_name] *= 100
            for name, cbardata in cbar_range.items():
                cbar_range[name] = (cbardata[0] * 100, cbardata[1] * 100)
            title += '(cm)'

        # --- 布局计算 (保持原逻辑) ---
        lon_range = self.extent[1] - self.extent[0]
        lat_range = self.extent[3] - self.extent[2]
        ratio = lat_range / lon_range
        subplot_width = 5
        subplot_height = subplot_width * ratio
        total_width = subplot_width * n_cols + 1
        total_height = subplot_height * n_rows + 1

        fig = plt.figure(figsize=(total_width, total_height), dpi=300)

        gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                      wspace=0.05, hspace=0.08,
                      width_ratios=[1, 1, 1, 1, 0.08],
                      height_ratios=[1] * n_rows)

        if 'data' in cbar_range:
            shared_cbar_range = cbar_range['data']
            use_shared = True
        else:
            use_shared = False
            shared_cbar_range = None

        axes = []
        pcm = None
        panel_idx = 0

        for i, (row_name, row_data) in enumerate(data.items()):
            if use_shared:
                current_cbar_range = shared_cbar_range
            else:
                current_cbar_range = cbar_range.get(row_name, cbar_range[list(cbar_range.keys())[0]])

            for j in range(n_cols):
                ax = self.init_ax(fig, gs[i, j])
                self.add_gridlines(ax, left=(j == 0), bottom=(i == n_rows - 1))

                # 1. 绘制背景色 (相关系数)
                pcm = self.plot_pcolor(
                    ax, row_data[j],
                    cmap, *current_cbar_range
                )

                # 2. 👈 绘制显著性斜线 (新增逻辑)
                if significance_data is not None and row_name in significance_data:
                    mask = significance_data[row_name][j]
                    if mask is not None:
                        # 使用 contourf 绘制斜线区域
                        # levels=[0.5, 1.5] 确保只捕捉值为 1 (True) 的区域
                        ax.contourf(
                            self.lon, self.lat, mask,
                            levels=[0.5, 1.5],
                            hatches=[hatch],
                            colors='none',      # 不填充颜色，只留斜线
                            edgecolor='none',
                            transform=ccrs.PlateCarree(),
                            zorder=3            # 确保在 pcolor 之上
                        )

                axes.append(ax)
                label = f"({string.ascii_lowercase[panel_idx]})"
                panel_idx += 1
                ax.set_title(label, fontsize=tick_labelsize + 4, pad=2)

            # 行标签 (保持原逻辑)
            first_ax_pos = axes[i * n_cols].get_position()
            fig.text(0.008, (first_ax_pos.y0 + first_ax_pos.y1) / 2,
                     row_name.replace('-None', '') if row_name != 'SV-Phys-None' else 'Phys-SV',
                     ha='left', va='center',
                     fontsize=row_label_fontsize, fontweight='bold')

        # --- 列标签和 Colorbar (保持原逻辑) ---
        for j in range(n_cols):
            ax = axes[j]
            pos = ax.get_position()
            x_center = (pos.x0 + pos.x1) / 2
            fig.text(x_center, 0.05, lead_labels[j], ha='center', va='bottom',
                     fontsize=tick_labelsize + 4, fontweight='bold')

        cbar_ax = fig.add_subplot(gs[:, n_cols])
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.set_label(f'{title}', fontsize=tick_labelsize + 2)
        cbar.ax.tick_params(labelsize=tick_labelsize)

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"OK 保存图：{save_path}")

