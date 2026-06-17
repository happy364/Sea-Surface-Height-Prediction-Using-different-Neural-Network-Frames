import numpy as np
from pathlib import Path
from mytools import MaskPearsonCorrNP, MSELossIgnoreNaN, MaskPearsonCorr
from configs import parse_args, get_my_config
import  torch

args = parse_args()
config = get_my_config(args)

mask_land = np.load(args.path_land_mask)
mask_valid = torch.from_numpy(~mask_land)

mse_func = MSELossIgnoreNaN(
                args, mask_valid, patched=False
            )
pcc_func = MaskPearsonCorr(
               mask_valid
            )
# all_models = ['SV', 'PR', 'SV+GC', 'PR+GC']
all_models = ['SV','PR','PF']
ensemble_members= {
    'Plan A':['SV', 'PR'],
    'Plan B':['SV+GC', 'PR+GC'],
    'Plan C':['SV','PR','SV+GC', 'PR+GC'],
}
ensemble_members = {
    'Plan A':['SV', 'PR'],
    'Plan B':['SV', 'PF'],
    'Plan C':['PF', 'PR'],
    'Plan D':['SV','PR','PF'],
}


# dir = Path('/data/hjj/ssh_prediction/work_dir/scs/ENSEMBLE_MEAN')
dir = Path('/data/hjj/ssh_prediction/work_dir/scs/parameter_3b_gc_0/pinn0_b4/full/spatial')
paths = {}
for model in all_models:
    paths[ model] = dir / f'{model}.npz'
# for pkl, model_name, folder in extract_model_info(dir):
#     paths[model_name] = pkl
results = {}
# rnn_ratio = torch.arange(0.1, 1.1, 0.1)  # [0.0, 0.1, 0.2, ..., 0.9]
# rnn_ratio = torch.tensor([0.3]*5+[0.7]*5)

# print(f"原始向量形状: {rnn_ratio.shape}")
# print(f"原始向量: {rnn_ratio}")
#
# # 在前面增加一个维度 (变成 2D)
# rnn_ratio = rnn_ratio.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(args.device)
# sv_ratio = (1-rnn_ratio).to(args.device)

for plan, members in ensemble_members.items():
    preds = 0
    nums = 0
    results[plan] = {}
    if nums == 0:
        targets = torch.from_numpy(np.load(paths[members[0]])['targets_spatial']).to(args.device).squeeze()

    save_path = dir/f'{ plan}.npz'
    for member in members:
        nums += 1
        path = paths[ member]
        pred = torch.from_numpy(np.load(path)['preds_spatial']).to(args.device).squeeze()
        if 'PR' in member:
            preds += pred
        else:
            preds += pred
    preds = preds/nums
    maes = torch.abs(preds - targets)
    mse = mse_func(preds.unsqueeze(0), targets.unsqueeze(0))
    rmse = torch.sqrt(mse)


    results['preds_spatial'] = preds.cpu().numpy()
    results['mae_spatial'] = maes.cpu().numpy()
    results['rmse'] = rmse.item()
    print(f'{plan} MAE: {maes[:,:,~mask_land].mean().item()*100:.2f}, RMSE: {rmse.item()*100:.2f}')
    np.savez(save_path, **results)

for model in all_models:
    data = np.load(paths[model])
    mae_spatial = data['mae_spatial'].squeeze()
    # print(f'mae shape: {mae_spatial.shape}')
    mae_mean = mae_spatial[:,:,~mask_land].mean()
    rmse = np.sqrt(np.square(mae_spatial[:,:,~mask_land]).mean())
    print(f'{model} MAE: {mae_mean*100:.2f}, RMSE: {rmse*100:.2f}')



