import os
import torch
import numpy as np
from pathlib import Path
import re

from tools.base_method import Model
from mytools import MSELossIgnoreNaN, set_all_seeds, MaskPearsonCorr
from configs import parse_args, get_my_config
from models import (
    PredFormer_Model,
    Mask_PredFormer_Model,
    SimVP_Model,
    RNN,
    ReST_Model,
    STED_Model
)
from dataset import MvDataset
from iterable_dataset import MakeIterDataset


class MultiModelEvaluator:

    def __init__(self, parent_dir, save_dir):
        self.parent_dir = parent_dir
        self.save_dir = save_dir
        self.results = {}

    # ---------------------------------------------------------
    # 1. 构造配置
    # ---------------------------------------------------------
    def build_args(self, filename, season='full'):
        args_ = parse_args()
        args_.need_ssh = True
        if  any(key in filename.lower() for key in ['mask', 'mi', 'phys']):
            args_.need_mask = True
        else:
            args_.need_mask = False

        args_.base = r"/data/hjj/SEJ/data/AVISO_0.125deg_indian_ocean" #todo
        time_ranges = {}
        time_ranges["full"] = ['2023-01-01','2026-06-14']
        time_ranges["Summer"] = ['2023-04-01','2023-09-30']
        time_ranges["Winter"] = ['2023-10-01','2024-03-31']
        time_ranges["1223"] = ['2023-12-23','2024-01-11']
        time_ranges["1224"] = ['2023-12-24','2024-01-12']
        time_ranges["0210"] = ['2023-02-10','2023-03-01']
        # if season is not None:
        #     args_.start_time_test, args_.end_time_test = time_ranges[season]

        prefix = filename.lower()

        # 定义前缀 → 模型名映射表
        MODEL_MAP = {
            "sv": "gsta",
            "tau": "tau",
            "pr": "predrnn",
            "pf": "predformer",
            "re": "rest",
        }

        # 匹配赋值
        for key, name in MODEL_MAP.items():
            if key in prefix:
                args_.model_name = name
                break

        args = get_my_config(args_)

        return args

    # ---------------------------------------------------------
    # 2. 构造模型
    # ---------------------------------------------------------
    def build_model(self, args, mask_land):

        if args.model_name == 'predformer':
            if args.mask_predformer:
                model = Mask_PredFormer_Model(args.model_config, mask_land)
            else:
                model = PredFormer_Model(args.model_config)
        elif args.model_name == 'gsta':
            model = SimVP_Model(**args.model_config)

        elif args.model_name == 'tau':
            model = SimVP_Model(**args.model_config)

        elif args.model_name == 'predrnn':
            model = RNN(args.model_config)
        elif args.model_name == 'rest':
            model = ReST_Model(**args.model_config)
        elif args.model_name == 'sted':
            model = STED_Model(**args.model_config)

        else:
            raise ValueError("Unsupported model")

        return model

    # ---------------------------------------------------------
    # 3. 加载统计量
    # ---------------------------------------------------------
    def load_statistics(self, args):

        stds = np.load(args.path_stds)
        means = np.load(args.path_means)
        adt_clim = torch.from_numpy(np.load(args.path_adt_clim))

        if stds.ndim == 0:
            std = stds.item()
            mean = means.item()
        else:
            std = stds[0]
            mean = means[0]

        return std, mean, adt_clim

    # ---------------------------------------------------------
    # 4. 单模型评估（lead + date）
    # ---------------------------------------------------------
    def evaluate_one_model(self, model, args, test_dataset,
                           mse_func, corr_func,
                           std, mean, adt_clim,
                           model_name):

        evaluator = Model(model, args)

        self.results[model_name] = {}

        # ---------- Lead ----------
        mse_list = evaluator.test_dataset(test_dataset, mse_func)
        rmse_list = torch.sqrt(mse_list)

        corr_list = evaluator.test_dataset(
            test_dataset,
            corr_func,
            is_per_sample=True,
            std_mean=(std, mean),
            multi_year_mean=adt_clim
        )

        self.results[model_name]['rmse_lead'] = \
            rmse_list.cpu().numpy() * std
        self.results[model_name]['corr_lead'] = \
            corr_list.cpu().numpy()

        # ---------- Per-Date ----------
        mse_date = evaluator.test_dataset(
            test_dataset, mse_func,
            is_per_date=True
        )
        rmse_date = torch.sqrt(mse_date)

        corr_date = evaluator.test_dataset(
            test_dataset,
            corr_func,
            is_per_date=True,
            is_per_sample=True,
            std_mean=(std, mean),
            multi_year_mean=adt_clim
        )

        self.results[model_name]['rmse_date'] = \
            rmse_date.cpu().numpy() * std
        self.results[model_name]['corr_date'] = \
            corr_date.cpu().numpy()

    # ---------------------------------------------------------
    # 5. Spatial 评估
    # ---------------------------------------------------------
    def evaluate_one_model_spatial(self, model, args, test_dataset,
                                   std, mean, model_name):

        evaluator = Model(model, args)

        self.results[model_name] = {}

        targets_per_date, preds_per_date = \
            evaluator.test_dataset_spatial(
                test_dataset,
                std_mean=(std, mean),
                is_per_date=True
            )

        self.results[model_name]['mae_spatial'] = \
            torch.abs(targets_per_date - preds_per_date).cpu().numpy()

        self.results[model_name]['targets_spatial'] = \
            targets_per_date.cpu().numpy()

        self.results[model_name]['preds_spatial'] = \
            preds_per_date.cpu().numpy()

    # ---------------------------------------------------------
    # 6. Persistence 评估
    # ---------------------------------------------------------
    def evaluate_persistence(self, evaluator, test_dataset,
                             mse_func, corr_func,
                             std, mean, adt_clim):

        model_name = "Persistence"
        self.results[model_name] = {}

        # Lead
        mse_list = evaluator.test_dataset(
            test_dataset,
            mse_func,
            is_persistence=True
        )
        rmse_list = torch.sqrt(mse_list)

        corr_list = evaluator.test_dataset(
            test_dataset,
            corr_func,
            is_per_sample=True,
            std_mean=(std, mean),
            multi_year_mean=adt_clim,
            is_persistence=True
        )

        # Date
        mse_date = evaluator.test_dataset(
            test_dataset,
            mse_func,
            is_per_date=True,
            is_persistence=True
        )
        rmse_date = torch.sqrt(mse_date)

        corr_date = evaluator.test_dataset(
            test_dataset,
            corr_func,
            is_per_date=True,
            is_per_sample=True,
            std_mean=(std, mean),
            multi_year_mean=adt_clim,
            is_persistence=True
        )

        self.results[model_name]['rmse_lead'] = \
            rmse_list.cpu().numpy() * std
        self.results[model_name]['corr_lead'] = \
            corr_list.cpu().numpy()
        self.results[model_name]['rmse_date'] = \
            rmse_date.cpu().numpy() * std
        self.results[model_name]['corr_date'] = \
            corr_date.cpu().numpy()

    # ---------------------------------------------------------
    # 7. 主运行函数
    # ---------------------------------------------------------
    def run(self, spatial=False, season='full'):

        model_info_list = extract_model_info(self.parent_dir)
        print(f"找到 {len(model_info_list)} 个模型文件，开始处理...")

        for model_para_path, simple_name, folder_path in model_info_list:
            print(f"*" * 50)
            print(f"\n处理模型: {simple_name}")
            print(f"模型路径: {model_para_path}")
            if not os.path.isfile(model_para_path):
                print(f"模型文件 {model_para_path} 不存在！")
                continue

            args = self.build_args(simple_name, season)
            set_all_seeds(args.SEED)

            mask_land = torch.from_numpy(
                np.load(args.path_land_mask)
            )

            mse_func = MSELossIgnoreNaN(
                args, ~mask_land, patched=False
            )

            corr_func = MaskPearsonCorr(~mask_land)

            model = self.build_model(args, mask_land)

            model.load_state_dict(
                torch.load(model_para_path, weights_only=True)
            )
            
            if season in ['Winter', 'Summer']:
                test_dataset = MakeIterDataset(args, mode='test', norm=True)
            else:
                test_dataset = MvDataset(args, mode='test', norm=True)

            std, mean, adt_clim = \
                self.load_statistics(args)
            model_name = simple_name+ f'-{season}' if season in ['Winter', 'Summer'] else simple_name
            if spatial:
                self.evaluate_one_model_spatial(
                    model, args, test_dataset,
                    std, mean,
                    model_name
                )
            else:
                self.evaluate_one_model(
                    model, args, test_dataset,
                    mse_func, corr_func,
                    std, mean, adt_clim,
                    model_name
                )

        # Persistence 单独计算（使用最后一个 args）
        if not spatial and len(self.results) > 0:
            evaluator = Model(model, args)
            self.evaluate_persistence(
                evaluator, test_dataset,
                mse_func, corr_func,
                std, mean, adt_clim
            )

        self.save_results(spatial= spatial)

    # ---------------------------------------------------------
    # 8. 保存
    # ---------------------------------------------------------
    def save_results(self, spatial=False):
        if spatial:
            save_dir = spatial_path = os.path.join(self.save_dir, "spatial")
            os.makedirs( spatial_path if spatial else self.save_dir, exist_ok=True)
        else:
            save_dir = self.save_dir

        for model_name, metrics in self.results.items():
            print(f"Saving {model_name}...")

            save_path = os.path.join(
               save_dir,
                f"{model_name}.npz"
            )

            np.savez(save_path, **metrics)

        print("Evaluation finished.")

def extract_model_info(parent_dir):
    """从父目录中提取所有 model_paras.pkl 文件的路径及相关信息"""
    parent_path = Path(parent_dir)
    pattern = re.compile(r'^(.*?)(?:_\d{8}_\d{4})?$')
    results = []

    for folder in parent_path.iterdir():
        if folder.is_dir():
            pkl_file = folder / "model_paras.pkl"
            if pkl_file.exists():
                name = folder.name
                match = pattern.match(name)
                simple_name = match.group(1) if match else name
                results.append((str(pkl_file), simple_name, str(folder)))
    return results

if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
        "expandable_segments:True"

    seasons = ["full", "Winter", "Summer","0210","1223","1224"]
    seasons= seasons[0:1]
    base_dir = Path(r"/data/hjj/ssh_prediction/work_dir/scs/seasonal_diff_spatial/other")
    for season in seasons:
        print(f"[Season: {season}]")
        save_dir = base_dir / season
        os.makedirs(save_dir, exist_ok=True)
        evaluator = MultiModelEvaluator(
            parent_dir=base_dir,
            save_dir= save_dir,
        )
        # evaluator.run(season= season)
        evaluator.run(spatial=True,season=season)