
import os
import xarray as xr
import dask.array as da
import  numpy as np
import re
from pathlib import Path
from configs import get_my_config, parse_args
args_ = parse_args()
args = get_my_config(args_)
xr.set_options(file_cache_maxsize=500)

# 假设输入的文件夹路径为 'input_folder'，输出文件夹路径为 'output_folder'
input_folder = Path(r"/data/hjj/SEJ/data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202411")
output_folder = args.base

os.makedirs(output_folder, exist_ok=True)

# 假设经纬度裁切范围为以下值 (根据实际需求修改)
lat_min, lat_max = -30, 30  # 纬度范围:
lon_min, lon_max = 40, 100  # 经度范围:


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

    adt = ds['adt'].astype(np.float32).sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    print( adt.dtype)
    ds_selected = xr.Dataset({
        "adt": adt,
    })

    # 保存结果
    output_path = output_folder / f"{year}.nc"
    ds_selected.to_netcdf(output_path)
    print(f"Saved data for {year} to {output_path}")


#
for year in range(1993,2025):
    print(f"processing {year}'s files")
    process_ym_data(year, input_folder, output_folder)
paths = sorted(output_folder.glob("*.nc"), key=lambda x: int(re.search(r'\d+', x.stem).group()))


# 按数字排序
# paths = sorted(paths, key=lambda x: int(re.search(r'\d+', x.stem).group()))

# 打开数据集（使用分块读取提高大文件处理效率）
data = xr.open_mfdataset(paths, chunks={'time': 100})  # 可根据实际情况调整chunks大小

# 提取训练数据时间段
train_data = data.sel(time=slice(args.start_time_train, args.end_time_train))

# 使用xarray内置方法计算统计量（自动跳过NaN）
# 这些计算是延迟执行的，只有需要时才实际计算
mean = train_data.adt.mean(skipna=True)  # 默认skipna=True
std = train_data.adt.std(skipna=True)

# 触发实际计算并将结果保存为标量值
mean_value = float(mean.values)
std_value = float(std.values)

print(f"mean: {mean_value}")
print(f"std: {std_value}")

# 保存统计量
np.save(args.path_means, mean_value)
np.save(args.path_stds, std_value)

# 计算陆地掩膜（使用第一时刻的数据）
# 更高效的方式：直接使用xarray操作
mask = train_data.adt.isel(time=0).isnull()  # isnull()在xarray中替代np.isnan()

# 保存掩膜
mask.values.tofile(args.path_land_mask)  # 或者用np.save
# 或者保持原来的保存方式
np.save(args.path_land_mask, mask.values)

# 或者更简洁的一步法
# np.save(args.path_land_mask, train_data.adt.isel(time=0).isnull().values)