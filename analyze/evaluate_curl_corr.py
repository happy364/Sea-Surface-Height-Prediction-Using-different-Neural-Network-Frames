import os
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
from plotter.spatial_plotter import PlotterMultiPanel
from mytools import compute_gradients_exact, MaskPearsonCorr


# ==============================
# 向量化相关系数
# ==============================
def calc_temporal_corr_p_vectorized(x, y, axis=0):
    N = x.shape[axis]

    x_mean = np.mean(x, axis=axis, keepdims=True)
    y_mean = np.mean(y, axis=axis, keepdims=True)

    x_dev = x - x_mean
    y_dev = y - y_mean

    cov = np.sum(x_dev * y_dev, axis=axis)
    var_x = np.sum(x_dev ** 2, axis=axis)
    var_y = np.sum(y_dev ** 2, axis=axis)

    denominator = np.sqrt(var_x * var_y)
    corr = np.divide(cov, denominator, out=np.zeros_like(cov), where=denominator != 0)

    df = N - 2
    r_clipped = np.clip(corr, -1.0 + 1e-8, 1.0 - 1e-8)
    t_stat = r_clipped * np.sqrt(df / (1.0 - r_clipped ** 2))
    p_val = 2 * stats.t.sf(np.abs(t_stat), df)

    return corr, p_val


# ==============================
# 主分析类
# ==============================
class WindStressAnalyzer:
    def __init__(self, args, dataset, pearson_tool, model_A, model_B=None):
        self.args = args
        self.dataset = dataset
        self.model_A = model_A
        self.model_B = model_B
        self.device = args.device
        self.pearson_tool = pearson_tool

    def _get_cd_parabolic(self, u_mag):
        u_mag_clipped = torch.clamp(u_mag, min=0.0)
        return (-0.0016 * (u_mag_clipped ** 2) + 0.0817 * u_mag_clipped + 0.484) / 1000.0

    def _calculate_curl(self, u10, v10):
        rho_air = 1.225
        u_mag = torch.sqrt(u10 ** 2 + v10 ** 2 + 1e-8)
        cd = self._get_cd_parabolic(u_mag)

        tau_x = rho_air * cd * u_mag * u10
        tau_y = rho_air * cd * u_mag * v10

        grad_tau_x_dy, _ = compute_gradients_exact(tau_x, self.dataset.lon, self.dataset.lat)
        _, grad_tau_y_dx = compute_gradients_exact(tau_y, self.dataset.lon, self.dataset.lat)

        return grad_tau_y_dx - grad_tau_x_dy

    # ==============================
    # 时间相关性 + 多面板绘图
    # ==============================
    def run_full_analysis_time(self, save_dir, mask_land):
        os.makedirs(save_dir, exist_ok=True)

        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False, num_workers=4)

        collected = {'err': [], 'curl': [], 'wind': [], 'imp': []}

        self.model_A.model.eval()
        if self.model_B:
            self.model_B.model.eval()

        print("Stage 1: Collecting data...")

        with torch.no_grad():
            for i, (datax, datay) in enumerate(dataloader):
                datax, datay = datax.to(self.device), datay.to(self.device)

                inputs = datax[:, :, :-2]
                pred_A = self.model_A.predict(inputs)

                err = torch.abs(pred_A[:, :, 0] - datay[:, :, 0])

                imp = None
                if self.model_B:
                    pred_B = self.model_B.predict(inputs)
                    err_B = torch.abs(pred_B[:, :, 0] - datay[:, :, 0])
                    imp = err - err_B

                wind = datay[:, :, -2:]
                u, v = wind[:, :, 0:1], wind[:, :, 1:2]

                curl = torch.abs(self._calculate_curl(u, v)).squeeze(2)
                wind_mag = torch.sqrt(u ** 2 + v ** 2).squeeze(2)

                collected['err'].append(err.cpu().numpy())
                collected['curl'].append(curl.cpu().numpy())
                collected['wind'].append(wind_mag.cpu().numpy())

                if imp is not None:
                    collected['imp'].append(imp.cpu().numpy())

        # 拼接
        err = np.concatenate(collected['err'], axis=0)
        curl = np.concatenate(collected['curl'], axis=0)
        wind = np.concatenate(collected['wind'], axis=0)
        imp = np.concatenate(collected['imp'], axis=0) if collected['imp'] else None

        print("Stage 2: Correlation...")

        corr_curl_err, p_curl_err = calc_temporal_corr_p_vectorized(curl, err)
        corr_wind_err, p_wind_err = calc_temporal_corr_p_vectorized(wind, err)

        if imp is not None:
            corr_curl_imp, p_curl_imp = calc_temporal_corr_p_vectorized(curl, imp)
            corr_wind_imp, p_wind_imp = calc_temporal_corr_p_vectorized(wind, imp)

        # ==============================
        # 多面板绘图
        # ==============================
        print("Stage 3: Multi-panel plotting...")

        lon_min, lon_max = np.min(self.dataset.lon), np.max(self.dataset.lon)
        lat_min, lat_max = np.min(self.dataset.lat), np.max(self.dataset.lat)
        extent = [lon_min, lon_max, lat_min, lat_max]

        plotter = PlotterMultiPanel(self.dataset.lon, self.dataset.lat, extent=extent)

        mask_valid = ~(mask_land.cpu().numpy() if torch.is_tensor(mask_land) else mask_land)

        target_leads = [0, 3, 6, 9]
        valid_leads = [t for t in target_leads if t < err.shape[1]]

        def prep(corr, p):
            maps = []
            for t in valid_leads:
                m = np.where(mask_valid, corr[t], np.nan)
                m[p[t] >= 0.05] = np.nan
                maps.append(m)
            return maps

        data = {
            "Err-Curl": prep(corr_curl_err, p_curl_err),
            "Err-Wind": prep(corr_wind_err, p_wind_err),
        }

        if imp is not None:
            data.update({
                "Imp-Curl": prep(corr_curl_imp, p_curl_imp),
                "Imp-Wind": prep(corr_wind_imp, p_wind_imp),
            })

        # colorbar
        all_vals = np.concatenate([np.ravel(v) for row in data.values() for v in row])
        vmax = np.nanmax(np.abs(all_vals))
        vmax = vmax if vmax > 0 else 1.0

        cbar_range = {'data': (-vmax, vmax)}
        lead_labels = [f"Day {t+1}" for t in valid_leads]

        save_path = os.path.join(save_dir, "multi_corr_maps.png")

        plotter.plot_panel(
            data=data,
            cbar_range=cbar_range,
            title="Correlation",
            save_path=save_path,
            cmap='RdBu_r',
            lead_labels=lead_labels
        )

        print(f"Saved: {save_path}")


# ==============================
# main
# ==============================
if __name__ == "__main__":
    from configs import parse_args, get_my_config
    from dataset import MvDataset
    from tools.base_method import Model
    from mytools import set_all_seeds
    import numpy as np
    from models import SimVP_Model

    args_ = parse_args()
    args_.need_wind = True
    args = get_my_config(args_)

    set_all_seeds(args.SEED)

    mask_land = torch.from_numpy(np.load(args.path_land_mask))

    modelA = SimVP_Model(**args.model_config)
    modelB = SimVP_Model(**args.model_config)

    stds = np.load(args.path_stds)
    model_para_path_A = r"/data/hjj/ssh_prediction/work_dir/scs/parameter_3b_gc_0/pinn0_b4/SV_20260103_1448/model_paras.pkl"
    model_para_path_B = r"/data/hjj/ssh_prediction/work_dir/scs/parameter_3b_gc_0.7/pinn_0.7_batchsize4/SV+GC_20251231_1236/model_paras.pkl"
    modelA.load_state_dict(
        torch.load(model_para_path_A, weights_only=True)
    )
    modelB.load_state_dict(
        torch.load(model_para_path_B, weights_only=True)
    )


    ModelA = Model(modelA, args)
    ModelB = Model(modelB, args)

    dataset = MvDataset(args, mode='test', norm=args.norm)

    pearson_tool = MaskPearsonCorr(~mask_land)

    analyzer = WindStressAnalyzer(args, dataset, pearson_tool, ModelA, ModelB)

    analyzer.run_full_analysis_time(
        save_dir="./results_corr",
        mask_land=mask_land
    )