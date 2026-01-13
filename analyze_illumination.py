#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
分析illumination prior和预测illumination的数值分布差异
"""

import os
import cv2
import yaml
import torch
import numpy as np
import torch.nn as nn
from os.path import join, isfile
from easydict import EasyDict
from torch.utils.data import DataLoader

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


def main():
    yaml_path = "/home/wzy/EvLight/options/test/sde_in.yaml"
    model_path = "/home/wzy/EvLight/log/train/sde_in/model_best.pth.tar"
    
    print("=" * 70)
    print("分析 Illumination Prior vs Predicted Illumination 的数值差异")
    print("=" * 70)
    
    # 加载配置
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    
    # 加载数据集
    _, val_dataset = get_dataset(config.DATASET)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # 加载模型
    model = get_model(config.MODEL)
    model = nn.DataParallel(model)
    model = model.cuda()
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    
    print("\n分析 i_24 序列的前5帧:\n")
    print("-" * 70)
    
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch["seq_name"][0] != "i_24":
                continue
            
            batch = move_tensors_to_cuda(batch)
            frame_id = batch["frame_id"][0]
            
            # 获取输入的亮度先验
            ill_prior = batch["ill_list"][0]
            
            # 构建单样本batch确保设备一致
            device = batch["lowligt_image"].device
            single_batch = {
                "lowligt_image": batch["lowligt_image"].to(device),
                "ill_list": [ill.to(device) for ill in batch["ill_list"]],
            }
            
            # 获取预测的illumination
            pred_ill, _ = model.module.IllumiinationNet(single_batch)
            
            # 计算预提亮图像
            low_img = batch["lowligt_image"]
            enhanced = low_img * pred_ill + low_img
            
            print(f"Frame: {frame_id}")
            print(f"  [亮度先验 Illumination Prior]")
            print(f"    min={ill_prior.min().item():.4f}, max={ill_prior.max().item():.4f}, "
                  f"mean={ill_prior.mean().item():.4f}, std={ill_prior.std().item():.4f}")
            
            print(f"  [预测的Illumination (pred_ill)]")
            print(f"    min={pred_ill.min().item():.4f}, max={pred_ill.max().item():.4f}, "
                  f"mean={pred_ill.mean().item():.4f}, std={pred_ill.std().item():.4f}")
            
            print(f"  [低光照图像 Low-light Image]")
            print(f"    min={low_img.min().item():.4f}, max={low_img.max().item():.4f}, "
                  f"mean={low_img.mean().item():.4f}")
            
            print(f"  [预提亮图像 Enhanced Image]")
            print(f"    min={enhanced.min().item():.4f}, max={enhanced.max().item():.4f}, "
                  f"mean={enhanced.mean().item():.4f}")
            
            # 计算增益因子
            gain = 1 + pred_ill
            print(f"  [增益因子 Gain = 1 + pred_ill]")
            print(f"    min={gain.min().item():.4f}, max={gain.max().item():.4f}, "
                  f"mean={gain.mean().item():.4f}")
            
            # 过曝像素比例
            overexposed = (enhanced > 1.0).float().mean().item() * 100
            print(f"  [过曝像素比例] {overexposed:.2f}%")
            
            print("-" * 70)
            
            count += 1
            if count >= 5:
                break
    
    print("\n结论分析:")
    print("=" * 70)
    print("1. 亮度先验范围在 [0, 1]，保持了原始图像的结构信息")
    print("2. 预测的illumination可能值域范围不合适（LeakyReLU无上界）")
    print("3. 如果增益因子过大，会导致图像过曝，细节丢失")
    print("4. IlluminationNet网络较浅，容易学到平滑的illumination map")


if __name__ == "__main__":
    main()

