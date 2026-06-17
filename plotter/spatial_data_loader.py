"""
空间数据准备模块 - 为 spatial_plotter.py 的 PlotterMultiPanel 准备数据

输入数据格式（实际从 analyze.py 得到）:
    npz 文件包含 'mae_spatial', 'preds_spatial', 'targets_spatial'
    原始 shape: (n_dates, 1, n_leads, 1, n_lat, n_lon)
    例如：(512, 1, 10, 1, 160, 160)
    squeeze 后：(n_dates, n_leads, n_lat, n_lon)

输出数据格式:
    符合 plot_panel(data, ...) 的格式：
    data = {
        '模型 1': [lead1_data, lead4_data, lead7_data, lead10_data],
        '模型 2': [lead1_data, lead4_data, lead7_data, lead10_data],
        ...
    }
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple


# ==================== 数据准备类 ====================

@dataclass
class SpatialDataLoader:
    """
    空间数据加载和预处理类

    用途:
        1. 从 npz 文件加载多模型数据
        2. 对时间维度平均 -> shape=(n_leads, n_lat, n_lon)
        3. 组织成 PlotterMultiPanel.plot_panel() 需要的格式

    输入数据格式:
        npz 文件包含 'mae_spatial', 'preds_spatial', 'targets_spatial'
        原始 shape: (n_dates, 1, n_leads, 1, n_lat, n_lon)

    属性:
        lon: 经度 (1D 或 2D)
        lat: 纬度 (1D 或 2D)
        lead_steps: 预测步长列表 [1,2,3,...,10]
        models: 存储加载的模型数据
    """
    lon: np.ndarray
    lat: np.ndarray
    lead_steps: List[int]
    models: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    def __post_init__(self):
        """确保 lon, lat 是 1D"""
        if self.lon.ndim == 2:
            self.lon = self.lon[0, :]
            self.lat = self.lat[:, 0]

    def load_models(self, model_files: Dict[str, str]):
        """
        加载多个模型的数据

        参数:
            model_files: {模型名：文件路径}

        每个 npz 文件应包含:
            - 'mae_spatial': shape=(n_dates, 1, n_leads, 1, n_lat, n_lon)
            - 'preds_spatial': shape=(n_dates, 1, n_leads, 1, n_lat, n_lon) [可选]
            - 'targets_spatial': shape=(n_dates, 1, n_leads, 1, n_lat, n_lon) [可选]
        """
        for name, filepath in model_files.items():
            print(f"加载模型：{name} 从 {filepath}")
            data = np.load(filepath)

            self.models[name] = {}
            for key in data.files:
                arr = data[key]
                print(f"  - {key}: shape={arr.shape}")

                if arr.ndim == 6:
                    arr = arr.squeeze(axis=(1, 3))
                    print(f"    squeeze 后：shape={arr.shape}")

                if arr.ndim == 4:
                    assert arr.shape[1] == len(self.lead_steps)
                    arr_mean = np.nanmean(arr, axis=0)
                    print(f"    时间平均后：shape={arr_mean.shape}")
                    self.models[name][key] = arr_mean
                else:
                    self.models[name][key] = arr

        print(f"OK 加载完成：{len(self.models)} 个模型")

    def get_model_ranking(
            self,
            metric_func,
            data_type: str = 'mae_spatial',
            leads: List[int] = None
    ) -> List[str]:
        """使用自定义函数对模型打分并排序"""
        if leads is None:
            leads = list(range(len(self.lead_steps)))

        scores = {}
        for model_name, model_data in self.models.items():
            if data_type not in model_data:
                continue
            arr = model_data[data_type][leads]
            # scores[model_name] = metric_func(arr) #todo
            if model_name == 'Target':
                scores[model_name]=0
                print(scores[model_name])
            elif model_name == 'SV+MI':
                scores[model_name]=2
            elif model_name == 'SV':
                scores[model_name]=1
            elif model_name == 'SV+GC':
                scores[model_name]=3
            elif model_name == 'SV-Phys':
                scores[model_name]=4
            else:
                scores[model_name]=5
            # scores[model_name] = metric_func(model_name)

        return sorted(scores.keys(), key=lambda k: scores[k])

    def prepare_data_for_panel(
            self,
            data_type: str = 'mae_spatial',
            leads: List[int] = [0, 3, 6, 9],
            include_target: bool = False,
            sort_models: bool = True, #todo
            sort_leads: List[int] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """
        增强版：支持模型排序输出
        """

        # 排序模型名
        model_names = list(self.models.keys())
        if sort_models:
            sorted_names = self.get_model_ranking(
                metric_func=lambda x: 1 ,
                data_type=data_type,
                leads=sort_leads or leads,

            )
            # 保留原始模型名，但按排序顺序重组
            data = {}
            for name in sorted_names:
                if name in model_names:
                    lead_data_list = [self.models[name][data_type][lead_idx] for lead_idx in leads]
                    data[name] = np.array(lead_data_list)
        else:
            data = {}
            for model_name in model_names:
                if data_type not in self.models[model_name]:
                    continue
                arr = self.models[model_name][data_type]
                data[model_name] = np.array([arr[lead_idx] for lead_idx in leads])

        # 添加 Target
        if include_target and 'targets_spatial' in list(self.models.values())[0]:
            first_key = sorted_names[0] if sort_models else list(data.keys())[0]
            target_mean = self.models[first_key].get('targets_spatial')
            if target_mean is not None:
                data['Target'] = np.array([target_mean[lead_idx] for lead_idx in leads])

        return data

    def get_cbar_range(self, data_type: str, leads: List[int], percentile: tuple = (1, 99)) -> Dict[str, tuple]:
        """
        自动计算 colorbar 范围

        参数:
            data_type: 数据类型
            leads: lead 索引列表
            percentile: 百分位范围
        """
        all_values = []
        for model_name, model_data in self.models.items():
            if data_type in model_data:
                arr = model_data[data_type]
                for lead_idx in leads:
                    all_values.append(arr[lead_idx])

        arr_combined = np.concatenate([arr.ravel() for arr in all_values])
        arr_combined = arr_combined[~np.isnan(arr_combined)]

        if len(arr_combined) == 0:
            return {'data': (0, 1)}

        vmin = np.nanpercentile(arr_combined, percentile[0])
        vmax = np.nanpercentile(arr_combined, percentile[1])

        return {'data': (vmin, vmax)}

    @classmethod
    def load_seasonal_model_diff(cls, lon, lat, lead_steps,
                                  model_a_winter_path: str,
                                  model_a_summer_path: str,
                                  model_b_winter_path: str,
                                  model_b_summer_path: str,
                                  model_a_name: str = 'Model-A',
                                  model_b_name: str = 'Model-B'):
        """
        加载数据并计算模型间差异（分季节）

        **正确的理解**:
        - 输入：4 个 MAE 文件 (winter-A, summer-A, winter-B, summer-B)
        - 计算:
          * Winter Diff: Winter-A - Winter-B (比较两个模型在冬季的表现)
          * Summer Diff: Summer-A - Summer-B (比较两个模型在夏季的表现)

        参数:
            lon, lat: 经纬度
            lead_steps: lead 步长列表
            model_a_winter_path: 模型 A 冬季数据路径
            model_a_summer_path: 模型 A 夏季数据路径
            model_b_winter_path: 模型 B 冬季数据路径
            model_b_summer_path: 模型 B 夏季数据路径
            model_a_name: 模型 A 的名称
            model_b_name: 模型 B 的名称

        返回:
            loader: SpatialDataLoader 对象，包含两个差异:
                - f'{model_a_name} - {model_b_name} (Winter)': Winter-A - Winter-B
                - f'{model_a_name} - {model_b_name} (Summer)': Summer-A - Summer-B

        数据形状:
            原始：[T, 10, 160, 160]
            减完：[T, 10, 160, 160]
            平均后：[10, 160, 160]
        """
        loader = cls(lon=lon, lat=lat, lead_steps=lead_steps)

        print("\n[加载并计算模型间差异 - 分季节]")

        # 加载所有 4 个文件
        files_to_load = {
            'winter_a': (model_a_winter_path, '模型 A 冬季'),
            'summer_a': (model_a_summer_path, '模型 A 夏季'),
            'winter_b': (model_b_winter_path, '模型 B 冬季'),
            'summer_b': (model_b_summer_path, '模型 B 夏季'),
        }

        loaded_data = {}
        for key, (filepath, desc) in files_to_load.items():
            print(f"  加载 {desc}: {filepath}")
            data = np.load(filepath)
            if 'mae_spatial' not in data.files:
                raise ValueError(f"文件 {filepath} 中没有 'mae_spatial' 键")

            mae = data['mae_spatial']
            if mae.ndim == 6:
                mae = mae.squeeze(axis=(1, 3))
            # 注意：这里**不立即对时间维度平均**
            loaded_data[key] = mae
            print(f"    原始 shape: {mae.shape}")

        # 分别计算冬季和夏季的差异
        print(f"\n[计算差异: {model_a_name} - {model_b_name}]")

        # 1. 冬季差异: Winter-A - Winter-B
        winter_diff = loaded_data['winter_a'] - loaded_data['winter_b']
        print(f"  Winter Diff (原始): {winter_diff.shape}")

        # 2. 夏季差异: Summer-A - Summer-B
        summer_diff = loaded_data['summer_a'] - loaded_data['summer_b']
        print(f"  Summer Diff (原始): {summer_diff.shape}")

        # 3. 对时间维度平均
        winter_diff_mean = np.nanmean(winter_diff, axis=0)  # [n_leads, n_lat, n_lon]
        summer_diff_mean = np.nanmean(summer_diff, axis=0)

        print(f"\n  Winter Diff (平均后): {winter_diff_mean.shape}")
        print(f"  Summer Diff (平均后): {summer_diff_mean.shape}")

        # 4. 存入 loader.models
        diff_name_winter = f'Winter'
        diff_name_summer = f'Summer'

        loader.models[diff_name_winter] = {'mae_spatial': winter_diff_mean}
        loader.models[diff_name_summer] = {'mae_spatial': summer_diff_mean}

        print(f"\n  OK {diff_name_winter}")
        print(f"  OK {diff_name_summer}")
        print(f"\nOK 加载季节模型差异完成：2 个对比 (Winter 和 Summer)")

        return loader


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("SpatialDataLoader 测试")

    # 测试 1: 单模型加载
    loader = SpatialDataLoader(
        lon=np.linspace(104, 124, 160),
        lat=np.linspace(2, 22, 160),
        lead_steps=list(range(1, 11))
    )

    if os.path.exists('SV.npz'):
        loader.load_models({'SV': 'SV.npz'})

        # 测试 2: 准备数据
        data = loader.prepare_data_for_panel('mae_spatial', leads=[0, 3, 6, 9])
        print(f"\n准备数据成功：{len(data)} 个模型")

        # 测试 3: 获取 colorbar 范围
        cbar_range = loader.get_cbar_range('mae_spatial', [0, 3, 6, 9])
        print(f"Colorbar 范围：{cbar_range}")

        print("\nOK 所有测试通过!")
    else:
        print("未找到 SV.npz，跳过测试")
