import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import math
from torch.amp import autocast, GradScaler
import time
import os
import logging
import gc
from abc import ABC, abstractmethod

def unpatchify_with_batch(patched_tensor, patch_size, original_channels):
    """
    对带 Batch 的 patchified tensor 进行还原，恢复为原始 (B, T, C, H, W) 格式。

    参数:
        patched_tensor: torch.Tensor，形状为 (B, T, C * p * p, H', W')
        patch_size: int，patch 大小 p
        original_channels: int，原始通道数 C

    返回:
        torch.Tensor，形状为 (B, T, C, H, W)
    """
    B, T, Cp2, H_, W_ = patched_tensor.shape
    p = patch_size
    C = original_channels

    assert Cp2 == C * p * p, f"通道数不匹配：{Cp2} != {C} * {p} * {p}"

    # 恢复 patch 通道维度为 patch 格式
    x = patched_tensor.reshape(B, T, C, p, p, H_, W_)

    # 交换维度：将 patch 中像素恢复到空间位置
    x = x.permute(0, 1, 2, 5, 3, 6, 4)  # (B, T, C, H', p, W', p)

    # 合并 patch 像素还原空间维度
    x = x.reshape(B, T, C, H_ * p, W_ * p)

    return x

class BaseMethod(ABC):
    def __init__(self, model,  config, loss_func_test=nn.MSELoss(), log_dir=None, optimizer=None, scheduler=None, mode='train'):
        self.config = config
        self.patched =  getattr(self.config, 'patched', False)
        self.patch_size = getattr(self.config, 'patch_size', None)
        self.output_channels = getattr(self.config, 'output_channels', 1)
        self.device = config.device
        self.model_config = config.model_config
        self.model = model.to(self.device)
        self.loss_name = getattr(self, 'loss_name', 'none')
        self.scaler = GradScaler()
        self.apply_early_stopping = True
        self.save_loss_curve = getattr(self.config, 'save_loss_curve', True)
        self.shuffle = getattr(self.config, 'shuffle', True)
        print(f"shuffle or not: {self.shuffle}")
        self.env = getattr(self.config, 'env', 'linux')
        self.loss_func_test = loss_func_test

        self.train_loss_history = []
        self.eval_loss_history = []

        if mode == 'train':
            self.log_dir = log_dir
            self.scheduler = scheduler
            self.optimizer = optimizer
            self.model_savepath = f"{log_dir}/model_paras.pkl"
            self._setup_logging()
            self._configure_gradient_clipping()
            self._configure_gradient_accumulation()
            self._log_config()
            self.logger.info("Model trainer initialized successfully")

    # ==================== 可复用的通用方法 ====================

    def _configure_gradient_clipping(self):
        """配置梯度裁剪参数 - 可复用"""
        self.gradient_clip_enabled = getattr(self.config, 'gradient_clip', False)
        self.gradient_clip_value = getattr(self.config, 'gradient_clip_value', 0.5)
        self.gradient_clip_type = getattr(self.config, 'gradient_clip_type', 'norm')

    def _clip_gradients(self):
        """执行梯度裁剪 - 可复用"""
        # # 检查梯度
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isinf(param.grad).any():
        #             print(f"INF gradient in {name}")
        if self.gradient_clip_enabled:
            if self.gradient_clip_type == 'norm':
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )
                if norm > 10:
                    self.logger.info(f"Extreme Gradient Norm: {norm:.2f}")
            elif self.gradient_clip_type == 'value':
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )

    def _configure_gradient_accumulation(self):
        """配置梯度累积参数 - 可复用"""
        if getattr(self.config, 'is_acc', False):
            self.accumulation_steps = max(1, self.config.acc_steps)
            self.batch_size_train = self.config.batch_size_train // self.accumulation_steps
            self.logger.info(f"Gradient accumulation enabled: "
                             f"Effective batch {self.config.batch_size_train} = "
                             f"actual batch {self.batch_size_train} × acc steps {self.accumulation_steps}")
        else:
            self.accumulation_steps = 1
            self.batch_size_train = self.config.batch_size_train
            self.logger.info(f"No gradient accumulation - Batch size: {self.batch_size_train}")


    def _cleanup_memory(self):
        """清理显存和内存 - 可复用"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _create_dataloader(self, dataset, is_train:bool = True, batch_size=None):
        """创建数据加载器 - 可复用"""
        if batch_size is None:
            batch_size = self.batch_size_train if is_train else max(self.config.batch_size_train, self.config.batch_size_eval)
        else:
            batch_size = batch_size
        num_workers = min(8, os.cpu_count() // 2) if self.env == 'linux' else 0
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.shuffle if is_train else False, #todo
            pin_memory=True,
            num_workers=num_workers,
        )

    def _setup_logging(self):
        """设置日志系统 - 可复用"""
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger('ModelTrainer')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            log_file = f"{self.log_dir}/training_{time.strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.log_file = log_file

    def _log_config(self):
        """记录模型和训练配置 - 可复用"""
        self.logger.info("=" * 60)
        self.logger.info("MODEL AND TRAINING CONFIGURATION")
        self.logger.info("=" * 60)

        self.logger.info("Model Configuration:")
        for key, value in self.model_config.items():
            self.logger.info(f"{key}: {value}")

        self.logger.info("Training Configuration:")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Model: {self.model.__class__.__name__}")
        self.logger.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        self.logger.info(f"  Scheduler: {self.scheduler.__class__.__name__}")
        self.logger.info(f"  Loss Function: {self.loss_name}")
        self.logger.info(f"  Batch Size: {self.batch_size_train}")
        self.logger.info(f"  Inputs: {'uv, ' if self.config.need_uv else ''} {'ssh, ' if self.config.need_ssh else ''}"
                         f" {'wind, 'if self.config.need_wind else ''} {'mask' if self.config.need_mask else ''}")
        self.logger.info(f"  pinn: {self.config.pinn_lambda}")
        self.logger.info(f"  norm: {self.config.norm}")
        self.logger.info(f"  need_wind: {self.config.need_wind}")

        if self.gradient_clip_enabled:
            self.logger.info(
                f"  Gradient Clipping: {self.gradient_clip_type} with value {self.gradient_clip_value}")
        else:
            self.logger.info("  Gradient Clipping: disabled")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        self.logger.info(f"  Total Parameters: {total_params:,}")
        self.logger.info(f"  Trainable Parameters: {trainable_params:,}")

        if torch.cuda.is_available():
            self.logger.info(
                f"  GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3:.1f}GB")
        self.logger.info("=" * 60)

    def _log_metrics(self, epoch, train_loss, eval_loss):
        """记录训练指标 - 可复用"""
        current_lr = self.scheduler.get_last_lr()[0]
        self.logger.info(f"Epoch {epoch + 1} Results: "
                         f"Train Loss={train_loss:.6f}, "
                         f"Eval Loss={eval_loss:.6f}, "
                         f"LR={current_lr:.2e}")

    def _early_stopping(self, eval_loss, best_loss, early_stop_counter, epoch):
        """
        处理早停逻辑 - 可复用
        :param eval_loss: 当前验证损失
        :param best_loss: 最佳验证损失
        :param early_stop_counter: 早停计数器
        :param epoch: 当前轮次
        :return: (best_loss, early_stop_counter, should_stop)
        """
        if eval_loss < best_loss:
            best_loss = eval_loss
            early_stop_counter = 0
            torch.save(self.model.state_dict(), self.model_savepath)
            self.logger.info(f"Model saved (New best loss: {best_loss:.6f})")
        else:
            early_stop_counter += 1
            self.logger.info(
                f"No improvement - Early stop counter: {early_stop_counter}/{self.config.patience}")
            if early_stop_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                return best_loss, early_stop_counter, True
        return best_loss, early_stop_counter, False

    # ==================== 训练流程的核心方法 ====================
    def train_model(self, train_dataset, eval_dataset, test_dataset=None):
        self.logger.info("Starting model training...")
        train_loader = self._create_dataloader(train_dataset, is_train=True)
        eval_loader = self._create_dataloader(eval_dataset, is_train=False)
        self.logger.info(f"Train dataset: {len(train_dataset)} samples")
        self.logger.info(f"Eval dataset: {len(eval_dataset)} samples")

        best_loss = math.inf
        early_stop_counter = 0
        start_time = time.time()

        self.logger.info(f"Training for {self.config.num_epochs} epochs "
                         f"with patience {self.config.patience}")

        for epoch in range(self.config.num_epochs):
            self.logger.info(f"{'=' * 20} Epoch {epoch + 1}/{self.config.num_epochs} {'=' * 20}")
            mask = self._create_mask(epoch)

            train_loss = self._train_epoch(train_loader, mask)
            eval_loss = self.test_model(eval_loader, self.mask_true_test if hasattr(self, 'mask_true_test') else None)
            self.train_loss_history.append(train_loss)
            self.eval_loss_history.append(eval_loss)
            self._log_metrics(epoch, train_loss, eval_loss)

            if self.config.model_name == 'predrnn':
                if epoch < self.model_config.r_sampling_step_2:
                    self.apply_early_stopping = False
                else:
                    self.apply_early_stopping = True

            if self.apply_early_stopping:
                self.scheduler.step(eval_loss)
                best_loss, early_stop_counter, should_stop = self._early_stopping(
                    eval_loss, best_loss, early_stop_counter, epoch)
                if should_stop:
                    break

        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.info(f"Training completed - Total time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
        self.logger.info(f"Best validation loss: {best_loss:.6f}")
        self._save_loss_curve()

        if test_dataset is not None:
            self.logger.info("Testing model on test dataset...")
            test_dataloader = self._create_dataloader(test_dataset, is_train=False)
            self.model.load_state_dict(torch.load(self.model_savepath))
            test_loss = self.test_model(test_dataloader, self.mask_true_test if hasattr(self, 'mask_true_test') else None)
            self.logger.info(f"Test loss: {test_loss:.6f}")


    def _train_epoch(self, dataloader, mask=None):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_time_start = time.time()

        self.logger.info(f"Starting epoch training with {len(dataloader)} batches")

        for step, train_data in enumerate(dataloader, 1):
            with autocast('cuda'):
                loss, batch_size = self._compute_loss(train_data, step, mask)
            if step == 1:
                max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
                self.logger.info(f"Max memory allocated: {max_memory:.2f}GB")
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            loss = loss / self.accumulation_steps
            self.scaler.scale(loss).backward()
            del loss
            if step % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                self._clip_gradients()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        # 把未完成的梯度累积处理掉
        if (step % self.accumulation_steps) != 0:
            self.scaler.unscale_(self.optimizer)
            self._clip_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        epoch_time = time.time() - epoch_time_start
        samples_per_sec = total_samples / epoch_time if epoch_time > 0 else 0
        self.logger.info(f"Epoch completed - Throughput: {samples_per_sec:.2f} samples/s")

        return math.sqrt(total_loss / total_samples) if total_samples > 0 else 0.0

    def test_model(self, dataloader, mask=None):
        """测试模型总体误差 - 可复用"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for step, eval_data in enumerate(dataloader, 1):
                loss, batch_size = self._compute_loss(eval_data, step, mask, test=True)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        return math.sqrt(total_loss / total_samples) if total_samples != 0 else 0.0

    def from_input_data(self, input_data):
        """
        Returns:
            inputs: Tensor
            targets: Tensor or None
        """

        if isinstance(input_data, (tuple, list)):
            inputs, targets = input_data

        elif isinstance(input_data, torch.Tensor):
            inputs = input_data

            if inputs.size(1) > self.config.input_length:
                targets = inputs[:, self.config.input_length:]
            else:
                targets = None
        else:
            raise ValueError("input_data must be tuple, list or torch.Tensor")

        inputs = inputs.to(self.device)

        if targets is not None:
            targets = targets.to(self.device)

        return inputs, targets

    def predict(self, input_data):

        self.model.eval()

        inputs, _ = self.from_input_data(input_data)

        with torch.no_grad():
            outputs = self.model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if self.patched:
                outputs = unpatchify_with_batch(
                    outputs,
                    self.patch_size,
                    self.output_channels
                )

        return outputs

    def test_dataset(self, dataset,  criterion=None, is_per_date=False, is_per_sample=False, is_persistence=False,
                        std_mean=None, multi_year_mean: torch.Tensor= None):
        """
        multi_year_mean: tensor, 3dims: c, h, w
        """

        criterion_per_lead = torch.zeros(self.config.output_length)
        dataloader = self._create_dataloader(dataset, is_train=False, batch_size=1 if is_per_date else None)
        criterion_per_date = []

        num = 0
        if criterion is None:
            criterion = self.loss_func_test

        for date_idx, eval_data in enumerate(dataloader):
            inputs, targets = self.from_input_data(eval_data)
            if self.patched:
                targets = unpatchify_with_batch(targets, self.patch_size, self.output_channels)

            if is_persistence:
                outputs = inputs[:, self.config.input_length-1].unsqueeze(1)
                if self.patched:
                    outputs = unpatchify_with_batch(outputs[:,:,:self.output_channels*self.patch_size**2], self.patch_size, self.output_channels)

                outputs = outputs[:,:,:self.output_channels].expand_as(targets)
            else:
                outputs = self.predict( inputs)
            if std_mean is not None:
                outputs, targets = self.denormalize(outputs, targets, std_mean)

            if multi_year_mean is not None:
                multi_year_mean = multi_year_mean.expand_as(targets).to(self.device)
                outputs = outputs - multi_year_mean
                targets = targets - multi_year_mean
            n = outputs.shape[0]
            num += n
            if is_per_date:
                criterion_per_date.append(0)
            for pred_idx in range(self.config.output_length):
                # print(f"outputs:{outputs.shape}, targets:{targets.shape}")
                if is_per_sample:
                    for batch_idx in range(n):
                        loss = criterion(outputs[batch_idx:batch_idx+1, pred_idx:pred_idx+1], targets[batch_idx:batch_idx+1, pred_idx:pred_idx+1])
                        criterion_per_lead[pred_idx] += loss.item()
                        if is_per_date:
                            criterion_per_date[date_idx] += loss.item()
                else:
                    loss = criterion(outputs[:, pred_idx:pred_idx+1], targets[:, pred_idx:pred_idx+1])
                    criterion_per_lead[pred_idx] += loss.item()*n
                    if is_per_date:
                        criterion_per_date[date_idx] += loss.item()*n

        if is_per_date:
            criterion_per_date = torch.tensor(criterion_per_date)
            criterion_per_date  /= self.config.output_length
            return criterion_per_date

        criterion_per_lead /= num
        return criterion_per_lead

    def test_dataset_spatial(self, dataset, std_mean= None, is_per_date=False, is_per_lead=False):
        dataloader = self._create_dataloader(dataset, is_train=False, batch_size=1 if is_per_date else None)
        if is_per_lead:
            MAE_per_lead = 0
            targets_per_lead = 0
            predictions_per_lead = 0
        elif is_per_date:
            targets_per_date = []
            predictions_per_date = []
        num = 0
        for date_idx, eval_data in enumerate(dataloader, 1):
            inputs, targets = self.from_input_data(eval_data)

            if self.patched:
                targets = unpatchify_with_batch(targets, self.patch_size, self.output_channels)
            outputs = self.predict( inputs)
            if std_mean is not None:
                outputs, targets = self.denormalize(outputs, targets, std_mean)
            n = outputs.shape[0]
            num += n

            if is_per_lead:
                MAE_per_lead += torch.nansum( torch.abs(outputs - targets), dim=0)
                targets_per_lead += torch.nansum( targets, dim=0)
                predictions_per_lead += torch.nansum( outputs, dim=0)
            elif is_per_date:
                targets_per_date.append(targets.cpu())
                predictions_per_date.append(outputs.cpu())

        if is_per_lead:
            MAE_per_lead /= num
            targets_per_lead /= num
            predictions_per_lead /= num
            return targets_per_lead, predictions_per_lead, MAE_per_lead
        elif is_per_date:
            targets_per_date = torch.stack(targets_per_date, dim=0)
            predictions_per_date = torch.stack(predictions_per_date, dim=0)
            return targets_per_date, predictions_per_date


    def denormalize(self, outputs, targets, std_mean):
        """
        Denormalize model outputs and targets.

        Assumptions
        -----------
        - Channel dimension is at axis = -3
          e.g.
            (B, C, H, W)
            (B, T, C, H, W)
            (C, H, W)

        Parameters
        ----------
        outputs : torch.Tensor
        targets : torch.Tensor
        std_mean : tuple or None
            (stds, means)
            - scalar (0-d): global normalization
            - 1-d (C,): channel-wise normalization
        """

        stds, means = std_mean

        # ---- convert to torch tensor ----
        stds = torch.as_tensor(stds, device=outputs.device, dtype=outputs.dtype)
        means = torch.as_tensor(means, device=outputs.device, dtype=outputs.dtype)

        # ---- case 1: scalar ----
        if stds.ndim == 0:
            outputs = outputs * stds + means
            targets = targets * stds + means
            return outputs, targets

        # ---- case 2: channel-wise ----
        if stds.ndim == 1:
            C = outputs.shape[-3]
            assert stds.numel() == C, \
                f"std channel mismatch: std {stds.numel()} vs data {C}"

            # build broadcast shape: (..., C, 1, 1)
            view_shape = [1] * outputs.ndim
            view_shape[-3] = C

            stds = stds.view(*view_shape)
            means = means.view(*view_shape)

            outputs = outputs * stds + means
            targets = targets * stds + means
            return outputs, targets

        # ---- unsupported ----
        raise ValueError(
            f"Unsupported std/mean dimension: std.ndim={stds.ndim}"
        )

    def _save_loss_curve(self):
        if not self.save_loss_curve:
            return

        try:
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.train_loss_history) + 1)

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, self.train_loss_history, label='Train Loss')
            plt.plot(epochs, self.eval_loss_history, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.title('Training & Validation Loss')
            plt.legend()
            plt.grid(True)

            save_path = os.path.join(self.log_dir, 'loss_curve.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

            self.logger.info(f"Loss curve saved to {save_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save loss curve: {e}")

    @abstractmethod
    def _compute_loss(self, train_data, step, mask=None, test=False):
        """
        Compute the loss for one batch.
        train_data: tuple of input data and target data or just input data (for RNN)
        step: train step in one epoch
        mask: needed if the model is RNN type, a tuple including train_mask and eval_mask.
        return： loss and the batch_size.
        """
        pass

    @abstractmethod
    def _create_mask(self, epoch):
        """
        :param epoch:
        :return: (mask_train and mask_eval) or None, depending on if needing mask or not.
        """
        pass

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self._cleanup_memory()
        except:
            pass

class Model(BaseMethod):
    def __init__(self, model, args):
        super().__init__(model, args,  mode="test")
    def _compute_loss(self,train_data, step, mask=None, test=False):
        pass
    def _create_mask(self,epoch):
        pass

class Model_Test(BaseMethod):
    def __init__(self, model, args, loss_func):
        super().__init__(model, args, loss_func_test=loss_func, mode="test")
    def _compute_loss(self,train_data, step, mask=None, test=False):
        pass
    def _create_mask(self,epoch):
        pass