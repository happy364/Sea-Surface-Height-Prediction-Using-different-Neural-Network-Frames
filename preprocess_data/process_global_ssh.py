
import os
import xarray as xr
import dask.array as da
import  numpy as np
import re
from pathlib import Path
from configs import get_my_config
args = get_my_config()
xr.set_options(file_cache_maxsize=500)

# 假设输入的文件夹路径为 'input_folder'，输出文件夹路径为 'output_folder'
input_folder = Path(r"/data/hjj/SEJ/data/MY_0.25/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_202411")
output_folder = args.base

os.makedirs(output_folder, exist_ok=True)

# 假设经纬度裁切范围为以下值 (根据实际需求修改)
lat_min, lat_max = 15, 45  # 纬度范围:
lon_min, lon_max = 110, 170  # 经度范围:


def process_ym_data(year, input_folder, output_folder):
    year_folder = input_folder / str(year)

    # 递归获取该年所有的 .nc 文件（月份文件夹下）
    files = sorted(year_folder.rglob("*.nc"), key=lambda x: int(re.search(r'\d+', x.stem).group()))

    if not files:
        print(f"No files found for year {year}.")
        return

    ds = xr.open_mfdataset(
        files,
        parallel=False,
        combine="by_coords",
        engine="netcdf4"
    )
    time  = ds['time']
    print(time[:10])

    # 剪裁经纬度范围
    print(ds['adt'].dtype)  # 查看原始数据类型

    adt = ds['adt'].astype(np.float32)#.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    print( adt.dtype)
    ds_selected = xr.Dataset({
        "adt": adt,
    })

    # 保存结果
    output_path = output_folder / f"{year}.nc"
    ds_selected.to_netcdf(output_path)
    print(f"Saved data for {year} to {output_path}")

#
#
# for year in range(1993,2026):
#     print(f"processing {year}'s files")
#     process_ym_data(year, input_folder, output_folder)
# paths = sorted(output_folder.glob("*.nc"), key=lambda x: int(re.search(r'\d+', x.stem).group()))
# 只处理包含数字的文件
paths = []
for file in output_folder.glob("*.nc"):
    if re.search(r'\d+', file.stem):
        paths.append(file)

# 按数字排序
paths = sorted(paths, key=lambda x: int(re.search(r'\d+', x.stem).group()))
data =  xr.open_mfdataset(paths)
# val_data  =  data.sel( time=slice(args.start_time_val, args.end_time_val))
# val_data.to_netcdf( output_folder / f"val.nc")
# del val_data
#
# test_data =  data.sel(time=slice(args.start_time_test, args.end_time_test))
# test_data.to_netcdf( output_folder / f"test.nc")
# del test_data

train_data = data.sel(time=slice(args.start_time_train, args.end_time_train))
del  data
# train_data.to_netcdf( output_folder / f"train.nc")


adt =  train_data.adt.values
del  train_data
mean =  np.nanmean(adt)
print(f"mean: {mean}")
std = np.nanstd(adt)
print(f"std: {std}")
np.save(args.path_means, mean)
np.save(args.path_stds, std)
mask =  np.isnan(adt[0])
np.save( args.path_land_mask, mask)