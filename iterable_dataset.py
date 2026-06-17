import numpy as np
import torch
from torch.utils.data import IterableDataset
import xarray as xr
import random
from collections import OrderedDict,defaultdict
import pandas as pd


class MakeIterDataset(IterableDataset):
    """
    online reading dataset
    """
    def __init__(self, args, mode='Train', norm=True):
        self.args = args
        self.mode = mode
        self.data = xr.open_mfdataset(self.args.datapath, combine='nested', concat_dim='time')

        if mode == 'train':
            time_span = slice(args.start_time_train, args.end_time_train)
        elif mode == 'eval':
            time_span = slice(args.start_time_val, args.end_time_val)
        elif mode == 'test':
            time_span = slice(args.start_time_test, args.end_time_test)
        elif mode == 'all':
            time_span = slice(args.start_time_train, args.end_time_test)
        else:
            raise ValueError('mode must be train, eval or test')

        self.data = self.data.sel(time=time_span)
        self.lat = self.data.latitude.values
        self.lon = self.data.longitude.values
        self.dates = self.data.time.values

        self.field = None

        if args.need_ssh:

            self.adt = np.expand_dims(self.data.adt.values.astype(np.float32), axis=1)
            if self.field is None:
                self.field = self.adt
            else:
                self.field = np.concatenate([self.field, self.adt], axis=1)
            # del self.adt
        if args.need_uv:
            self.uo = self.data.ugos.values.astype(np.float32)
            self.vo = self.data.vgos.values.astype(np.float32)
            self.field = np.concatenate([self.uo, self.vo], axis=1)
            # del self.uo, self.vo

        # ssh data and ugos, vgos were not been  normalized
        self.mean = np.load(args.path_means)
        self.std = np.load(args.path_stds)
        if norm:
            print(self.field.shape)
            print(self.mean.shape)
            print(f"mean: {self.mean}, std: {self.std}")
            for i in range(self.field.shape[1]):
                if isinstance(self.mean, (int, float, np.number)):
                    self.field[:, i, ...] = (self.field[:, i, ...] - self.mean) / self.std
                elif self.mean.ndim == 0:
                    self.field[:, i, ...] = (self.field[:, i, ...] - self.mean) / self.std
                else:
                    self.field[:, i, ...] = (self.field[:, i, ...] - self.mean[i]) / self.std[i]

        if args.need_wind:
            self.wind_data = xr.open_dataset(args.wind_path).sel(time=time_span)
            self.u10 = np.expand_dims(self.wind_data.u10.values.astype(np.float32), axis=1)
            self.v10 = np.expand_dims(self.wind_data.v10.values.astype(np.float32), axis=1)
            self.field = np.concatenate([self.field, self.u10, self.v10], axis=1)
            del self.u10, self.v10

        if args.need_mask:
            mask = np.load(args.path_land_mask)
            mask = mask[None, None, ...]
            mask = np.tile(mask, (self.field.shape[0], 1, 1, 1))
            self.field = np.concatenate([self.field, mask], axis=1)
            del mask

        self.field[abs(self.field) > 100] = np.nan

        # mask_ssh = np.isnan(self.adt[0, 0])
        # np.save('mask.npy', mask_ssh)

        self.input_length = args.input_length
        self.field = np.nan_to_num(self.field)

        self.output_length = args.output_length
        if args.patched:
            p = self.args.model_config['patch_size']
            T = self.field.shape[0]
            C = self.field.shape[1]
            H, W = self.field.shape[2:]
            # 先 reshape 出补丁维度
            self.field = self.field.reshape(T, C, H // p, p, W // p, p)
            # 交换维度使得每个 patch 内的像素连续
            self.field = self.field.transpose(0, 1, 3, 5, 2, 4)
            # 再 reshape 合并 patch 内像素到通道维度
            self.field = self.field.reshape(T, C * p * p, H // p, W // p)

        self.field = torch.from_numpy(self.field)
        print(f"field shape: {self.field.shape}")

        st_min = self.input_length
        ed_max = self.field.shape[0] - self.output_length
        self.time_indices = list(range(st_min, ed_max+1))
        self.length = len(self.time_indices)

        if self.args.winter_only or  self.args.summer_only:
            # 1. 用一个临时 dict 收集原始的跨年段
            raw = defaultdict(list)
            for idx, dt in enumerate(self.dates):
                m, y = dt.month, dt.year
                if not self.args.summer_only and not self.args.winter_only:
                    raw[y].append(idx)
                elif self.args.summer_only:
                    if 4 <= m <= 9:
                        raw[y].append(idx)
                else:  # winter_only
                    if m >= 10:
                        raw[y].append(idx)
                    elif m <= 3:
                        raw[y - 1].append(idx)

            # 2. 过滤掉「不完整」的跨年冬季段
            #    只保留那些同时有 (year,10–12) 和 (year+1,1–3) 两部分的 season_year
            self.segments = OrderedDict()
            if mode == 'train' and args.winter_only:
                self.segments[1992] = raw[1992]

            for season_year in sorted(raw):
                idxs = raw[season_year]
                if self.args.winter_only:
                    # 拆两部分看有没有数据
                    oct_dec = [i for i in idxs if (
                            self.dates[i].year == season_year and 10 <= self.dates[i].month <= 12)]
                    jan_mar = [i for i in idxs if (
                            self.dates[i].year == season_year + 1 and 1 <= self.dates[i].month <= 3)]
                    if not (oct_dec and jan_mar):
                        continue  # 不完整，剔除
                    combined = sorted(oct_dec + jan_mar)
                    self.segments[season_year] = combined

                elif self.args.summer_only:
                     self.segments[season_year] = sorted(idxs)

                else:
                    # 全年模式下直接按年保留
                    self.segments[season_year] = sorted(idxs)

            # 3. 统计总长度（实际样本数要减去前后 input/output 长度）
            self.length = sum(
                max(0, len(idxs) - self.input_length - self.output_length)
                for idxs in self.segments.values()
            )
            self.field = torch.from_numpy(self.field).to(self.args.device)

    def __iter__(self):
        if self.args.winter_only or  self.args.summer_only:
            years = list(self.segments.keys())
            # 如果是训练集，打乱年序
            if self.mode == 'train':
                random.shuffle(years)
            for season_year in years:
                indices = self.segments[season_year]
                indices = indices[self.input_length:len(indices)-self.output_length]
                if self.mode == 'train':
                    random.shuffle(indices)
                for indice in indices:
                    dataX = torch.nan_to_num(self.field[indice - self.input_length  : indice])
                    dataY = self.field[indice  : indice + self.output_length,0:self.args.output_channel,...]

                    yield dataX, dataY
        else:

            if self.mode == 'train':
                random.shuffle(self.time_indices)
            for indice in self.time_indices:
                dataX = torch.from_numpy(np.nan_to_num(self.field[indice - self.input_length: indice]))
                dataY = torch.from_numpy(self.field[indice: indice + self.output_length,0:self.args.output_channel, ...])
                yield dataX, dataY

    def __len__(self):
        return self.length
    def get_grids(self):
        return  self.lon, self.lat

    def get_info(self):
        return self.mean, self.std
    def get_times(self):
        return self.dates[self.time_indices[0]: self.time_indices[-1]+1]


if __name__ == '__main__':
    from configs import mypara
    evalset = MakeIterDataset(mypara,'test')
    print(f'len {len(evalset)}')
    dates = evalset.get_times()
    print(len( dates))
    print(dates[0])
    print(dates[-1])
    n=0
    for dataX, dataY in evalset:
        n+=1
    print(f'{n} n')

