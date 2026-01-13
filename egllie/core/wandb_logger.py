"""
WandB Logger for EvLight0-2
===========================

提供 WandB 集成功能，用于可视化训练过程中的：
1. 标量指标: loss, PSNR, PSNR*, SSIM
2. 图像可视化: 预提亮中间结果 (L_coarse, L_refined, R_prior_raw, R_pre)
"""

import os
import torch
import numpy as np
from typing import Dict, Optional, List, Any
from absl.logging import info, warning


# 检查 wandb 是否可用
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warning("wandb not installed. Install with: pip install wandb")


class WandbLogger:
    """
    WandB 日志记录器
    
    功能:
        1. 记录标量指标 (loss, PSNR, PSNR*, SSIM)
        2. 可视化图像 (输入/输出/中间结果)
        3. 支持训练和验证阶段的分别记录
        4. 固定样本可视化 (观察同一图像在不同训练阶段的变化)
    """
    
    def __init__(
        self,
        project: str = "EvLight0-2",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        save_dir: Optional[str] = None,
        enabled: bool = True,
        log_images_every: int = 100,  # 每多少步记录一次图像
        max_images_per_log: int = 4,   # 每次最多记录几张图像
    ):
        # 固定样本缓存 (用于观察同一图像在不同训练阶段的变化)
        self._fixed_samples = None
        self._fixed_samples_set = False
        """
        初始化 WandB Logger
        
        Args:
            project: WandB 项目名称
            run_name: 运行名称 (可选，自动生成)
            config: 配置字典，用于记录超参数
            save_dir: 本地保存目录
            enabled: 是否启用 wandb
            log_images_every: 每多少步记录一次图像
            max_images_per_log: 每次最多记录几张图像
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.log_images_every = log_images_every
        self.max_images_per_log = max_images_per_log
        self._step = 0
        
        if not self.enabled:
            if enabled and not WANDB_AVAILABLE:
                warning("WandB logging requested but wandb not available")
            return
        
        # 初始化 wandb
        wandb.init(
            project=project,
            name=run_name,
            config=config,
            dir=save_dir,
            reinit=True,
        )
        info(f"WandB initialized: project={project}, run={wandb.run.name}")
    
    def log_scalars(
        self,
        scalars: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        记录标量指标
        
        Args:
            scalars: 标量字典 {name: value}
            step: 全局步数 (可选)
            prefix: 前缀 (如 "train/" 或 "val/")
        """
        if not self.enabled:
            return
        
        log_dict = {}
        for name, value in scalars.items():
            key = f"{prefix}{name}" if prefix else name
            log_dict[key] = value
        
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
    
    def log_metrics(
        self,
        losses: Dict[str, float],
        metrics: Dict[str, float],
        epoch: int,
        phase: str = "train"
    ):
        """
        记录训练/验证指标
        
        Args:
            losses: 损失字典 {loss_name: value}
            metrics: 指标字典 {metric_name: value}
            epoch: 当前 epoch
            phase: "train" 或 "val"
        """
        if not self.enabled:
            return
        
        log_dict = {"epoch": epoch}
        
        # 记录损失
        for name, value in losses.items():
            log_dict[f"{phase}/loss/{name}"] = value
        
        # 记录指标 (PSNR, PSNR*, SSIM)
        for name, value in metrics.items():
            log_dict[f"{phase}/metric/{name}"] = value
        
        wandb.log(log_dict, step=epoch)
    
    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        epoch: int,
        batch_idx: int = 0,
        phase: str = "train",
        captions: Optional[Dict[str, str]] = None
    ):
        """
        记录图像可视化
        
        Args:
            images: 图像字典 {name: tensor}, tensor shape (B, C, H, W)
            epoch: 当前 epoch
            batch_idx: batch 索引
            phase: "train" 或 "val"
            captions: 图像标题字典
        """
        if not self.enabled:
            return
        
        # 控制图像记录频率
        global_step = epoch * 10000 + batch_idx
        if global_step % self.log_images_every != 0 and phase == "train":
            return
        
        log_dict = {}
        captions = captions or {}
        
        for name, tensor in images.items():
            if tensor is None:
                continue
            
            # 确保 tensor 在 CPU 上
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu()
            
            # 取前几个样本
            if tensor.dim() == 4:  # (B, C, H, W)
                num_samples = min(tensor.shape[0], self.max_images_per_log)
                wandb_images = []
                
                for i in range(num_samples):
                    img = tensor[i]  # (C, H, W)
                    img_np = self._tensor_to_numpy(img)
                    caption = captions.get(name, name)
                    wandb_images.append(wandb.Image(img_np, caption=f"{caption}_{i}"))
                
                log_dict[f"{phase}/image/{name}"] = wandb_images
            
            elif tensor.dim() == 3:  # (C, H, W)
                img_np = self._tensor_to_numpy(tensor)
                caption = captions.get(name, name)
                log_dict[f"{phase}/image/{name}"] = wandb.Image(img_np, caption=caption)
        
        if log_dict:
            wandb.log(log_dict, step=epoch)
    
    def log_preenhance_intermediates(
        self,
        intermediates: Dict[str, torch.Tensor],
        low_light_img: torch.Tensor,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor,
        epoch: int,
        batch_idx: int = 0,
        phase: str = "train",
        total_batches: int = 1697
    ):
        """
        记录预提亮中间结果可视化
        
        专门用于可视化:
        - L_coarse: 粗光照图
        - L_refined: 精细光照图  
        - R_prior_raw: 逐元素除法结果
        - R_pre: 预提亮图像结果
        
        Args:
            intermediates: 中间结果字典，包含上述键
            low_light_img: 低光输入图像 (B, 3, H, W)
            pred_img: 预测输出图像 (B, 3, H, W)
            gt_img: Ground truth 图像 (B, 3, H, W)
            epoch: 当前 epoch
            batch_idx: batch 索引
            phase: "train" 或 "val"
            total_batches: 每个 epoch 的总 batch 数
        """
        if not self.enabled:
            return
        
        # 使用 global_step 使不同阶段的图像都能保存
        global_step = epoch * total_batches + batch_idx
        
        # 控制图像记录频率
        if phase == "train" and batch_idx % self.log_images_every != 0:
            return
        
        log_dict = {}
        num_samples = min(low_light_img.shape[0], self.max_images_per_log)
        
        # 基础图像
        base_images = {
            "input_lowlight": low_light_img,
            "output_pred": pred_img,
            "gt": gt_img,
        }
        
        # 预提亮中间结果
        preenhance_keys = ["L_coarse", "L_refined", "R_prior_raw", "R_pre", "pred_illumaintion"]
        
        for key in preenhance_keys:
            if key in intermediates and intermediates[key] is not None:
                base_images[key] = intermediates[key]
        
        # 转换并记录图像
        for name, tensor in base_images.items():
            if tensor is None:
                continue
            
            tensor = tensor.detach().cpu()
            wandb_images = []
            
            for i in range(num_samples):
                if tensor.dim() == 4:
                    img = tensor[i]
                else:
                    img = tensor
                
                # 光照图特殊处理 (单通道 -> 归一化显示)
                if name in ["L_coarse", "L_refined", "pred_illumaintion"]:
                    img_np = self._illumination_to_numpy(img)
                elif name == "R_prior_raw":
                    # 逐元素除法结果可能有极端值，需要裁剪
                    img_np = self._tensor_to_numpy(img, clip_range=(-1, 2))
                else:
                    img_np = self._tensor_to_numpy(img)
                
                # 添加 epoch 和 batch 信息到 caption
                wandb_images.append(wandb.Image(
                    img_np, 
                    caption=f"ep{epoch}_b{batch_idx}_{name}_{i}"
                ))
            
            log_dict[f"{phase}/preenhance/{name}"] = wandb_images
        
        # 使用 global_step 记录，这样可以在 wandb 上通过 slider 查看不同阶段
        if log_dict:
            wandb.log(log_dict, step=global_step)
    
    def set_fixed_samples_from_dataset(self, dataset, num_samples: int = 5):
        """
        从数据集中均匀采样固定样本，用于观察同一图像在不同训练阶段的变化
        
        Args:
            dataset: 训练数据集
            num_samples: 固定样本数量 (默认 5)
        """
        if self._fixed_samples_set:
            return  # 只设置一次
        
        dataset_size = len(dataset)
        # 计算均匀分布的采样索引
        indices = [int(i * dataset_size / num_samples) for i in range(num_samples)]
        info(f"Sampling fixed samples at indices: {indices} (dataset size: {dataset_size})")
        
        # 收集样本
        samples_list = []
        for idx in indices:
            sample = dataset[idx]
            samples_list.append(sample)
        
        # 合并为 batch 格式
        self._fixed_samples = {}
        
        # 堆叠各个字段
        self._fixed_samples['lowligt_image'] = torch.stack([s['lowligt_image'] for s in samples_list])
        self._fixed_samples['normalligt_image'] = torch.stack([s['normalligt_image'] for s in samples_list])
        self._fixed_samples['event_free'] = torch.stack([s['event_free'] for s in samples_list])
        self._fixed_samples['lowlight_image_blur'] = torch.stack([s['lowlight_image_blur'] for s in samples_list])
        
        # 处理 ill_list (列表中的每个元素需要堆叠)
        if 'ill_list' in samples_list[0]:
            num_ill = len(samples_list[0]['ill_list'])
            self._fixed_samples['ill_list'] = [
                torch.stack([s['ill_list'][i] for s in samples_list])
                for i in range(num_ill)
            ]
        
        # 保存采样的索引信息
        self._fixed_sample_indices = indices
        self._fixed_samples_set = True
        info(f"Fixed {num_samples} uniformly distributed samples for visualization tracking")
    
    def set_fixed_samples(self, batch: Dict[str, torch.Tensor], num_samples: int = 4):
        """
        从 batch 中设置固定样本 (备用方法)
        
        Args:
            batch: 包含图像数据的 batch 字典
            num_samples: 固定样本数量
        """
        if self._fixed_samples_set:
            return  # 只设置一次
        
        self._fixed_samples = {}
        n = min(num_samples, batch['lowligt_image'].shape[0])
        
        # 深拷贝固定样本到 CPU
        self._fixed_samples['lowligt_image'] = batch['lowligt_image'][:n].detach().cpu().clone()
        self._fixed_samples['normalligt_image'] = batch['normalligt_image'][:n].detach().cpu().clone()
        self._fixed_samples['event_free'] = batch['event_free'][:n].detach().cpu().clone()
        self._fixed_samples['lowlight_image_blur'] = batch['lowlight_image_blur'][:n].detach().cpu().clone()
        
        # 复制 ill_list
        if 'ill_list' in batch:
            self._fixed_samples['ill_list'] = [ill[:n].detach().cpu().clone() for ill in batch['ill_list']]
        
        self._fixed_samples_set = True
        info(f"Fixed {n} samples for visualization tracking")
    
    def get_fixed_samples(self, device='cuda'):
        """
        获取固定样本 (移动到指定设备)
        
        Args:
            device: 目标设备
        
        Returns:
            固定样本 batch 字典，如果未设置则返回 None
        """
        if not self._fixed_samples_set or self._fixed_samples is None:
            return None
        
        batch = {}
        for key, val in self._fixed_samples.items():
            if key == 'ill_list':
                batch[key] = [ill.to(device) for ill in val]
            else:
                batch[key] = val.to(device)
        return batch
    
    def log_fixed_samples_progress(
        self,
        model,
        epoch: int,
        global_step: int,
        device: str = 'cuda'
    ):
        """
        使用固定样本进行推理并记录可视化，观察训练进度
        
        Args:
            model: 训练中的模型
            epoch: 当前 epoch
            global_step: 全局递增的 step (WandB 要求单调递增)
            device: 计算设备
        """
        if not self.enabled or not self._fixed_samples_set:
            return
        
        # 获取固定样本
        batch = self.get_fixed_samples(device)
        if batch is None:
            return
        
        # 模型推理 (eval 模式)
        # 注意：使用位置参数而非 kwargs，避免 DataParallel + batch_size<GPU数 时的 bug
        model.eval()
        with torch.no_grad():
            outputs = model(batch, True)  # True = return_intermediates
        model.train()
        
        # 记录图像
        log_dict = {}
        num_samples = batch['lowligt_image'].shape[0]
        
        images_to_log = {
            "input": batch['lowligt_image'],
            "gt": batch['normalligt_image'],
            "pred": outputs['pred'],
        }
        
        # 添加中间结果
        if 'intermediates' in outputs:
            for key in ['L_coarse', 'L_refined', 'R_prior_raw', 'R_pre', 'pred_illumaintion']:
                if key in outputs['intermediates'] and outputs['intermediates'][key] is not None:
                    images_to_log[key] = outputs['intermediates'][key]
        
        # 转换并记录
        for name, tensor in images_to_log.items():
            if tensor is None:
                continue
            
            tensor = tensor.detach().cpu()
            wandb_images = []
            
            for i in range(num_samples):
                img = tensor[i] if tensor.dim() == 4 else tensor
                
                if name in ["L_coarse", "L_refined", "pred_illumaintion"]:
                    img_np = self._illumination_to_numpy(img)
                elif name == "R_prior_raw":
                    img_np = self._tensor_to_numpy(img, clip_range=(-1, 2))
                else:
                    img_np = self._tensor_to_numpy(img)
                
                wandb_images.append(wandb.Image(img_np, caption=f"sample{i}_ep{epoch}"))
            
            log_dict[f"fixed_samples/{name}"] = wandb_images
        
        if log_dict:
            wandb.log(log_dict, step=global_step)
    
    def _tensor_to_numpy(
        self,
        tensor: torch.Tensor,
        clip_range: tuple = (0, 1)
    ) -> np.ndarray:
        """
        将 tensor 转换为 numpy 数组 (用于 wandb.Image)
        
        Args:
            tensor: (C, H, W) 或 (H, W) tensor
            clip_range: 裁剪范围
        
        Returns:
            numpy array (H, W, C) 或 (H, W), 值域 [0, 255]
        """
        if tensor.dim() == 3:  # (C, H, W)
            img = tensor.permute(1, 2, 0).numpy()  # (H, W, C)
        else:  # (H, W)
            img = tensor.numpy()
        
        # 裁剪到指定范围
        img = np.clip(img, clip_range[0], clip_range[1])
        
        # 归一化到 [0, 1]
        if clip_range != (0, 1):
            img = (img - clip_range[0]) / (clip_range[1] - clip_range[0])
        
        # 转换到 [0, 255]
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def _illumination_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将光照图 tensor 转换为可视化的 numpy 数组
        
        光照图通常是单通道，需要归一化并转为灰度图显示
        
        Args:
            tensor: (1, H, W) 或 (H, W) tensor
        
        Returns:
            numpy array (H, W), 值域 [0, 255]
        """
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)  # (H, W)
        
        img = tensor.numpy()
        
        # 归一化到 [0, 1]
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-6:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        
        # 转换到 [0, 255]
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """
        监控模型梯度和参数
        
        Args:
            model: PyTorch 模型
            log_freq: 记录频率
        """
        if not self.enabled:
            return
        wandb.watch(model, log="all", log_freq=log_freq)
    
    def finish(self):
        """结束 wandb 运行"""
        if self.enabled:
            wandb.finish()
            info("WandB run finished")


def create_wandb_logger(config, save_dir: str) -> WandbLogger:
    """
    从配置创建 WandB Logger
    
    Args:
        config: EasyDict 配置对象
        save_dir: 保存目录
    
    Returns:
        WandbLogger 实例
    """
    # 检查配置中是否启用 wandb
    wandb_config = getattr(config, 'WANDB', None)
    
    if wandb_config is None:
        # 默认配置
        return WandbLogger(
            project="EvLight0-2",
            run_name=None,
            config=dict(config) if hasattr(config, 'items') else None,
            save_dir=save_dir,
            enabled=True,
            log_images_every=100,
            max_images_per_log=2,
        )
    
    return WandbLogger(
        project=wandb_config.get('project', 'EvLight0-2'),
        run_name=wandb_config.get('run_name', None),
        config=dict(config) if hasattr(config, 'items') else None,
        save_dir=save_dir,
        enabled=wandb_config.get('enabled', True),
        log_images_every=wandb_config.get('log_images_every', 100),
        max_images_per_log=wandb_config.get('max_images_per_log', 2),
    )

