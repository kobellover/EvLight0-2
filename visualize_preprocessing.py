#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
可视化EvLight预处理部分的中间结果:
1. 亮度先验 (illumination prior)
2. 初步提亮的图片 (preliminary enhanced image)
3. SNR图 (Signal-to-Noise Ratio map)

针对SDE_indoor测试集的i_24序列
"""

import os
import cv2
import yaml
import torch
import numpy as np
import torch.nn as nn
from os.path import join, isfile
from collections import OrderedDict
from easydict import EasyDict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from egllie.datasets import get_dataset
from egllie.models import get_model


def move_tensors_to_cuda(dictionary_of_tensors):
    if isinstance(dictionary_of_tensors, dict):
        return {
            key: move_tensors_to_cuda(value)
            for key, value in dictionary_of_tensors.items()
        }
    if isinstance(dictionary_of_tensors, torch.Tensor):
        return dictionary_of_tensors.cuda(non_blocking=True)
    else:
        return dictionary_of_tensors


def save_image(tensor, path, normalize=False):
    """保存张量为图像"""
    if not isinstance(tensor, torch.Tensor):
        return
    image = tensor.detach()
    image = image.permute(1, 2, 0).cpu().numpy()
    if normalize:
        if image.shape[-1] == 1:
            image_single = np.squeeze(image, axis=-1)
        else:
            image_single = image
        image_max = np.max(image_single)
        image_min = np.min(image_single)
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min + 0.0001)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def save_colormap(tensor, path, cmap='jet'):
    """保存单通道张量为伪彩色图像"""
    if not isinstance(tensor, torch.Tensor):
        return
    image = tensor.detach().squeeze().cpu().numpy()
    # 归一化到 [0, 1]
    image_min = np.min(image)
    image_max = np.max(image)
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)
    
    # 使用matplotlib colormap
    colormap = plt.get_cmap(cmap)
    colored = colormap(image)[:, :, :3]  # 去掉alpha通道
    colored = (colored * 255).astype(np.uint8)
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, colored)


class PreprocessVisualizer:
    """预处理可视化器 - 提取和保存中间结果"""
    
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_preprocessing(self, batch, seq_filter="i_24"):
        """
        可视化预处理阶段的中间结果
        """
        with torch.no_grad():
            B = batch["lowligt_image"].shape[0]
            
            for b in range(B):
                seq_name = batch["seq_name"][b]
                frame_id = batch["frame_id"][b]
                
                # 只处理指定序列
                if seq_name != seq_filter:
                    continue
                
                print(f"Processing: {seq_name}/{frame_id}")
                
                # 创建保存目录
                seq_dir = join(self.save_dir, seq_name)
                os.makedirs(seq_dir, exist_ok=True)
                
                # 提取单个样本的batch (保持在同一设备上)
                device = batch["lowligt_image"].device
                single_batch = {
                    "lowligt_image": batch["lowligt_image"][b:b+1].to(device),
                    "normalligt_image": batch["normalligt_image"][b:b+1].to(device),
                    "event_free": batch["event_free"][b:b+1].to(device),
                    "lowlight_image_blur": batch["lowlight_image_blur"][b:b+1].to(device),
                    "ill_list": [ill[b:b+1].to(device) for ill in batch["ill_list"]],
                }
                
                # === 1. 获取亮度先验 (输入的illumination prior) ===
                input_illumination_prior = batch["ill_list"][0][b]  # 输入的亮度先验
                
                # === 2. 通过IlluminationNet获取预测的illumination ===
                pred_illumination, illu_feature = self.model.module.IllumiinationNet(single_batch)
                single_batch["illumaintion"] = pred_illumination
                single_batch["illu_feature"] = illu_feature
                
                # === 3. 计算初步提亮的图片 ===
                low_light_img = single_batch["lowligt_image"]
                low_light_img_blur = single_batch["lowlight_image_blur"]
                enhance_low_img_mid = low_light_img * pred_illumination + low_light_img
                enhance_low_img_blur = low_light_img_blur * pred_illumination + low_light_img_blur
                
                # === 4. 计算SNR图 ===
                snr_map = self._snr_generate(enhance_low_img_mid, enhance_low_img_blur, snr_factor=3.0)
                
                # === 只保存对比图 ===
                self._save_comparison(
                    low_img=batch["lowligt_image"][b],
                    gt_img=batch["normalligt_image"][b],
                    ill_prior=input_illumination_prior,
                    pred_ill=pred_illumination[0],
                    enhanced_img=enhance_low_img_mid[0],
                    snr_map=snr_map[0],
                    save_path=join(seq_dir, f"{frame_id}_comparison.png"),
                    frame_id=frame_id
                )
                
                print(f"  Saved visualizations for {frame_id}")
    
    def _snr_generate(self, low_img, low_img_blur, snr_factor=3.0):
        """生成SNR图"""
        # 转换为灰度图
        dark = low_img
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = low_img_blur
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        
        mask = torch.div(light, noise + 0.0001).contiguous()
        
        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        
        mask = mask * snr_factor / (mask_max + 0.0001)
        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        return mask
    
    def _save_comparison(self, low_img, gt_img, ill_prior, pred_ill, enhanced_img, snr_map, save_path, frame_id):
        """保存对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Preprocessing Visualization - Frame: {frame_id}', fontsize=14)
        
        # 1. 低光照图像
        low_np = low_img.permute(1, 2, 0).cpu().numpy()
        low_np = np.clip(low_np, 0, 1)
        axes[0, 0].imshow(low_np)
        axes[0, 0].set_title('Low-light Image')
        axes[0, 0].axis('off')
        
        # 2. Ground Truth
        gt_np = gt_img.permute(1, 2, 0).cpu().numpy()
        gt_np = np.clip(gt_np, 0, 1)
        axes[0, 1].imshow(gt_np)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # 3. 亮度先验 (输入)
        ill_prior_np = ill_prior.squeeze().cpu().numpy()
        im3 = axes[0, 2].imshow(ill_prior_np, cmap='hot')
        axes[0, 2].set_title('Illumination Prior (Input)')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
        
        # 4. 预测的illumination
        pred_ill_np = pred_ill.squeeze().cpu().numpy()
        im4 = axes[1, 0].imshow(pred_ill_np, cmap='hot')
        axes[1, 0].set_title('Predicted Illumination')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
        
        # 5. 初步提亮的图像
        enhanced_np = enhanced_img.permute(1, 2, 0).cpu().numpy()
        enhanced_np = np.clip(enhanced_np, 0, 1)
        axes[1, 1].imshow(enhanced_np)
        axes[1, 1].set_title('Preliminary Enhanced Image')
        axes[1, 1].axis('off')
        
        # 6. SNR图
        snr_np = snr_map.squeeze().cpu().numpy()
        im6 = axes[1, 2].imshow(snr_np, cmap='jet')
        axes[1, 2].set_title('SNR Map')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    # 配置
    yaml_path = "/home/wzy/EvLight/options/test/sde_in.yaml"
    model_path = "/home/wzy/EvLight/log/train/sde_in/model_best.pth.tar"
    save_dir = "/home/wzy/EvLight/visualization_results/preprocessing_i24"
    
    print("=" * 60)
    print("EvLight 预处理可视化脚本")
    print("=" * 60)
    print(f"配置文件: {yaml_path}")
    print(f"模型路径: {model_path}")
    print(f"保存目录: {save_dir}")
    print("=" * 60)
    
    # 加载配置
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    
    # 加载数据集 (只需要验证集)
    print("\n[1/4] 加载数据集...")
    _, val_dataset = get_dataset(config.DATASET)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    print(f"  验证集大小: {len(val_dataset)}")
    
    # 加载模型
    print("\n[2/4] 加载模型...")
    model = get_model(config.MODEL)
    model = nn.DataParallel(model)
    model = model.cuda()
    
    if not isfile(model_path):
        raise ValueError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    print("  模型加载成功!")
    
    # 创建可视化器
    print("\n[3/4] 初始化可视化器...")
    visualizer = PreprocessVisualizer(model, save_dir)
    
    # 进行可视化
    print("\n[4/4] 开始可视化预处理结果...")
    print("  只处理 i_24 序列")
    print("-" * 40)
    
    processed_count = 0
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            # 只处理 i_24 序列
            if batch["seq_name"][0] != "i_24":
                continue
            
            batch = move_tensors_to_cuda(batch)
            visualizer.visualize_preprocessing(batch, seq_filter="i_24")
            processed_count += 1
            
            # 可以设置限制处理的帧数
            # if processed_count >= 10:
            #     break
    
    print("-" * 40)
    print(f"\n完成! 共处理 {processed_count} 帧")
    print(f"结果保存在: {save_dir}")
    print("\n可视化结果包括:")
    print("  1. *_1_low_light.png          - 原始低光照图像")
    print("  2. *_2_ground_truth.png       - Ground Truth (正常光照)")
    print("  3. *_3_illumination_prior.png - 亮度先验 (输入)")
    print("  4. *_4_pred_illumination.png  - 预测的illumination map")
    print("  5. *_5_preliminary_enhanced.png - 初步提亮的图片")
    print("  6. *_6_snr_map.png            - SNR图")
    print("  7. *_comparison.png           - 对比图")


if __name__ == "__main__":
    main()

