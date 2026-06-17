import numpy as np
import csv
from mytools import MaskPearsonCorrNP
from configs import parse_args, get_my_config

args = parse_args()
config = get_my_config(args)

mask_land = np.load(args.path_land_mask)
pcc = MaskPearsonCorrNP(~mask_land)
mask_valid = ~mask_land

model_names = ['SV', 'PR', 'PF']
seasons = ['Summer', 'Winter']

# 统一结果容器（扁平结构，方便导出 CSV）
results = []

for model_name in model_names:
    for season in seasons:
        path = f'/data/hjj/ssh_prediction/work_dir/scs/parameter_3b_gc_0/pinn0_b4/{season}/spatial/{model_name}-{season}.npz'
        data = np.load(path)

        # -------- 构造 target（不存 dict，直接局部变量）--------
        t0 = data['targets_spatial'].squeeze()[:, 0]
        t1 = data['targets_spatial'].squeeze()[-1, 1:]
        target = np.concatenate([t0, t1], axis=0)  # (N, H, W)
        print(f'target shape: {target.shape}')

        # -------- 计算统计量 --------
        std_field = target.std(axis=0)
        mean_std_field = std_field.mean()          # 记录1

        mean_target = target.mean(axis=0)
        std_mean_target = mean_target.std()        # 记录2

        mae_field = np.mean(data['mae_spatial'].squeeze(), axis=(0, 1))
        rmse = np.sqrt(np.mean(np.square(data['mae_spatial'].squeeze()[:,:,mask_valid])))
        corr_val, p_value = pcc(std_field, mae_field, return_pvalue=True)       # 记录3

        # -------- 存结果 --------
        results.append({
            'model': model_name,
            'season': season,
            'mean_std_field': round(float(mean_std_field)*100, 2),
            'std_mean_target': round(float(std_mean_target)*100,2),
            'corr_std_mae': round(float(corr_val),2),
            'p_value': float(p_value)*100,
            'rmse': round(float(rmse)*100,2)
        })

# -------- 写入 CSV --------
csv_path = './work_dir/summary_metrics.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['model', 'season', 'mean_std_field', 'std_mean_target', 'corr_std_mae', 'p_value', 'rmse']
    )
    writer.writeheader()
    writer.writerows(results)

print(f'Saved to {csv_path}')