"""
EvLight 数据集模块 - 事件数据预处理与Voxel Grid生成

该文件负责:
===========
1. 加载事件相机数据和低光照/正常光照图像对
2. 将原始事件流转换为Voxel Grid表示
3. 生成亮度先验 (Illumination Prior)
4. 数据增强 (裁剪等)

Voxel Grid 表示:
================
事件相机输出的是异步事件流，每个事件包含 (t, x, y, p):
- t: 时间戳
- x, y: 像素坐标  
- p: 极性 (+1 表示亮度增加, -1 表示亮度减少)

Voxel Grid 将这些异步事件转换为规则的3D张量:
- 维度: [C, H, W]
- C 是时间通道数 (默认32)，表示将时间段等分为C个bin
- 每个bin累积该时间段内的事件极性

这种表示方式使得事件数据可以直接输入卷积神经网络处理!
"""

import numpy as np
import os
from torch.utils.data import Dataset, ConcatDataset
import cv2
import random
import torch


class eglol_withNE_dataset(Dataset):
    """
    EvLight 低光照数据集类 (with Normal Events)
    
    功能:
    =====
    1. 加载低光照图像和对应的正常光照图像(GT)
    2. 加载并处理事件数据，生成Voxel Grid
    3. 生成亮度先验 (Illumination Prior)
    4. 训练时随机裁剪，测试时中心裁剪
    
    数据组织结构:
    =============
    dataset_root/
    ├── sequence_1/
    │   ├── normal/              # 正常光照图像
    │   │   ├── 00001.png
    │   │   ├── ...
    │   │   └── normalight_event.npz  # 正常光照下的事件
    │   └── low/                 # 低光照图像
    │       ├── 00001.png
    │       ├── ...
    │       └── lowlight_event.npz    # 低光照下的事件
    ├── sequence_2/
    └── ...
    
    参数:
    =====
    dataset_root: 数据集根目录
    height, width: 原始图像尺寸
    seq_name: 序列名称
    is_train: 是否训练模式
    voxel_grid_channel: Voxel Grid的时间通道数 (默认32)
    is_split_event: 是否使用分割后的事件文件
    """
    def __init__(
        self,
        dataset_root,
        height,
        width,
        seq_name,
        is_train,
        voxel_grid_channel,
        is_split_event
    ):
        self.H = height
        self.W = width
        
        # ==================== 加载正常光照图像列表 ====================
        self.noraml_img_folder = os.path.join(dataset_root, seq_name, "normal")
        self.noraml_img_list_all = os.listdir(self.noraml_img_folder)
        # 过滤出png图像文件
        self.noraml_img_list = sorted(list(filter(lambda x: 'png' in x, self.noraml_img_list_all)))
        
        # ==================== 加载低光照图像列表 ====================
        self.low_img_folder = os.path.join(dataset_root, seq_name, "low")
        self.low_img_list_all = os.listdir(self.low_img_folder)
        self.low_img_list = sorted(list(filter(lambda x: 'png' in x, self.low_img_list_all)))

        # ==================== 事件数据配置 ====================
        self.is_split_event = is_split_event
        if self.is_split_event:
            # 使用分割后的事件文件 (每帧一个npz)
            self.noraml_ev_list = sorted(list(filter(lambda x: 'npz' in x, self.noraml_img_list_all)))[:-1]
            self.low_ev_list = sorted(list(filter(lambda x: 'npz' in x, self.low_img_list_all)))[:-1]   
        else:
            # 使用完整的事件文件 (整个序列一个npz)
            self.normal_event_file = os.path.join(
                dataset_root, seq_name, "normal", "normalight_event.npz"
            )
            self.low_event_file = os.path.join(
                dataset_root, seq_name, "low", "lowlight_event.npz"
            )
                 
        self.num_input = len(self.low_img_list)
        self.ev_idx = None
        self.events = None
        
        # ==================== 裁剪参数 ====================
        self.center_cropped_height = 256
        self.random_cropped_width = 256
        self.is_train = is_train
        self.seq_name = seq_name
        
        # ==================== Voxel Grid配置 ====================
        # voxel_grid_channel: 时间维度的通道数，将时间段等分为这么多个bin
        # 默认32通道，意味着将两帧之间的事件按时间分成32个时间片
        self.voxel_grid_channel = voxel_grid_channel

    def __len__(self):
        return self.num_input

    def get_event(self, idx):
        """
        从完整事件文件中提取当前帧对应的事件
        
        逻辑:
        =====
        对于第idx帧图像，提取从前一帧时间戳到后一帧时间戳之间的所有事件
        
        Args:
            idx: 帧索引
            
        Returns:
            event: 事件数组 [N, 4]，每行是 (timestamp, x, y, polarity)
        """
        # 确定事件的起始时间戳
        if idx == 0:
            # 第一帧: 从第一个事件开始
            start_t = self.events[0, 0]
        else:
            # 其他帧: 从前一帧的时间戳开始
            start_t = int(self.low_img_list[idx - 1][:-4])
            
        # 确定事件的结束时间戳
        if idx == self.num_input - 1:
            # 最后一帧: 到最后一个事件结束
            end_t = self.events[-1, 0]
        else:
            # 其他帧: 到下一帧的时间戳结束
            end_t = int(self.low_img_list[idx + 1][:-4])
            
        # 提取时间范围内的事件
        ev_start_idx_list = np.where(self.events[:, 0] > start_t)
        ev_start_idx = ev_start_idx_list[0][0]
        ev_end_idx_list = np.where(self.events[:, 0] < end_t)
        ev_end_idx = ev_end_idx_list[0][-1]
        event = self.events[ev_start_idx:ev_end_idx]

        return event

    def _crop(self, input_frame_list, events_list):
        """
        裁剪图像和事件数据
        
        训练模式: 随机裁剪 (数据增强)
        测试模式: 中心裁剪 (保证一致性)
        
        Args:
            input_frame_list: 图像列表 [low_img, gt_img, illumination_map, blur_img]
            events_list: 事件数据列表
            
        Returns:
            crop_image_list: 裁剪后的图像列表
            output_events_list: 裁剪后的事件列表
        """
        # ==================== 确定裁剪区域 ====================
        if self.is_train:
            # 训练时随机裁剪位置
            min_y = random.randint(0, self.W - self.random_cropped_width)
            min_x = random.randint(0, self.H - self.center_cropped_height)
        else:
            # 测试时使用中心裁剪
            min_y = (self.W - self.random_cropped_width) // 2
            min_x = (self.H - self.center_cropped_height) //2
        max_y = min_y + self.random_cropped_width
        max_x = min_x + self.center_cropped_height
        
        # ==================== 裁剪图像 ====================
        crop_image_list = []
        for input_frame in input_frame_list:
            input_frames = input_frame[min_x:max_x, min_y:max_y, :]
            # 转换为PyTorch格式: [H, W, C] → [C, H, W]
            input_frames_torch = torch.from_numpy(input_frames).permute(2,0,1).float() 
            crop_image_list.append(input_frames_torch)
        
        # ==================== 裁剪事件 ====================
        # 事件裁剪需要同时更新坐标
        output_events_list = []
        for events in events_list:
            # 筛选x坐标在裁剪范围内的事件
            mask_x = torch.where(
                (events[:, 2] < max_x) & (events[:, 2] >= min_x)
            )
            event_x = torch.index_select(events, 0, mask_x[0])
            
            # 筛选y坐标在裁剪范围内的事件
            mask_y = torch.where(
                (event_x[:, 1] < max_y) & (event_x[:, 1] >= min_y)
            )
            event_y = torch.index_select(event_x, 0, mask_y[0])
            event = event_y.clone()
            
            # 更新坐标: 减去裁剪区域的偏移量
            event[:, 2] = event_y[:, 2] - min_x  # x坐标
            event[:, 1] = event_y[:, 1] - min_y  # y坐标
            output_events_list.append(event)
            
        return crop_image_list, output_events_list

    def _illumiantion_map(self, img):
        """
        生成亮度先验 (Illumination Prior)
        
        计算方法:
        =========
        取RGB三通道的逐像素最大值
        
        原理:
        =====
        在低光照图像中，RGB通道的最大值可以近似反映场景的照明情况
        这个先验用于IlluminationNet的输入，帮助网络估计光照分布
        
        Args:
            img: RGB图像 [H, W, 3]
            
        Returns:
            initial_illumination_map: 亮度先验 [H, W]
        """
        intial_illumiantion_map = np.max(img, axis=2)
        return intial_illumiantion_map

    def _generate_voxel_grid(self, event):
        """
        将事件流转换为Voxel Grid表示
        
        ============================================================
        这是EvLight中事件特征提取的起点! 
        Voxel Grid是事件数据进入神经网络的标准表示形式
        ============================================================
        
        Voxel Grid原理:
        ===============
        事件相机产生的是异步事件流 (t, x, y, p)
        为了能用CNN处理，需要转换为规则的张量格式
        
        Voxel Grid将时间轴等分为C个bin (默认32个):
        - 每个事件根据其时间戳分配到对应的时间bin
        - 每个bin累积该时间段内所有事件的极性值
        
        生成的Voxel Grid: [C, H, W]
        - C: 时间通道数 (voxel_grid_channel, 默认32)
        - H, W: 空间分辨率
        - 每个位置的值: 该时间bin、该像素位置的事件极性累积和
        
        极性处理:
        =========
        原始极性 p ∈ {0, 1} 被转换为 {-1, +1}:
        - p=0 → -1 (亮度减少)
        - p=1 → +1 (亮度增加)
        
        Args:
            event: 事件张量 [N, 4]
                   每行: (timestamp, x, y, polarity)
                   
        Returns:
            voxel_grid: Voxel Grid [C, H, W]
                        C=voxel_grid_channel (默认32)
        
        示例:
        =====
        假设 voxel_grid_channel=32, 图像尺寸 256x256
        输入: 10000个事件，时间跨度 0.01秒
        输出: [32, 256, 256] 的张量
        
        每个通道代表 0.01/32 ≈ 0.3ms 的时间段内的事件累积
        """
        width, height = self.random_cropped_width, self.center_cropped_height
        
        # ==================== Step 1: 时间归一化 ====================
        # 将事件时间戳归一化到 [0, voxel_grid_channel-1]
        event_start = event[0, 0]    # 第一个事件的时间戳
        event_end = event[-1, 0]     # 最后一个事件的时间戳
        
        # 计算每个事件所属的时间通道 (bin)
        # ch = (t - t_start) / (t_end - t_start) * C
        ch = (
            event[:, 0].to(torch.float32)
            / (event_end - event_start)
            * self.voxel_grid_channel
        ).long()
        
        # 确保通道索引在有效范围内
        torch.clamp_(ch, 0, self.voxel_grid_channel - 1)
        
        # ==================== Step 2: 提取坐标和极性 ====================
        ex = event[:, 1].long()  # x坐标
        ey = event[:, 2].long()  # y坐标
        ep = event[:, 3].to(torch.float32)  # 极性
        
        # 极性转换: {0, 1} → {-1, +1}
        # p=0 表示亮度减少，转为 -1
        # p=1 表示亮度增加，保持 +1
        ep[ep == 0] = -1

        # ==================== Step 3: 构建Voxel Grid ====================
        # 初始化空的Voxel Grid
        voxel_grid = torch.zeros(
            (self.voxel_grid_channel, height, width), dtype=torch.float32
        )
        
        # 使用index_put_将事件累积到对应位置
        # accumulate=True 表示同一位置的多个事件会累加
        # voxel_grid[ch, ey, ex] += ep
        voxel_grid.index_put_((ch, ey, ex), ep, accumulate=True)

        return voxel_grid
    
    def __getitem__(self, index):
        """
        获取单个训练样本
        
        Returns:
            sample: 字典，包含:
                - lowligt_image: 低光照图像 [3, H, W]
                - normalligt_image: 正常光照图像(GT) [3, H, W]
                - event_free: 事件Voxel Grid [32, H, W]
                - lowlight_image_blur: 模糊的低光照图像 [3, H, W]
                - ill_list: 亮度先验列表 [[1, H, W]]
                - seq_name: 序列名称
                - frame_id: 帧ID
        """
        # ==================== 1. 加载事件数据 ====================
        if (self.events is None) & (self.is_split_event == False):
            # 从完整事件文件加载
            events = np.load(self.low_event_file)
        elif self.is_split_event == True:
            # 从分割事件文件加载
            events = np.load(os.path.join(self.low_img_folder, self.low_ev_list[index]))
        else:
            raise ValueError('w/o assign event')

        try:
            self.events = events["arr_0"]
            # 处理结构化数组格式
            if self.events.ndim == 1:
                et = self.events["timestamp"]
                ex = self.events["x"]
                ey = self.events["y"]
                ep = self.events["polarity"]
                self.events = np.stack([et, ex, ey, ep], axis=1)
        except:
            print(f"loading event error @ index: {index}")


        if self.is_split_event == False:
            try:
                event_input = self.get_event(index)
            except:
                print(f"loading event error @ seq: {self.seq_name}")
        else:
            event_input = self.events
        
        del events

        # ==================== 2. 加载图像并生成亮度先验 ====================
        # 加载低光照图像
        img_low = cv2.cvtColor(
            cv2.imread(
                os.path.join(self.low_img_folder, self.low_img_list[index])
            ),
            cv2.COLOR_BGR2RGB,
        )
        
        # 生成模糊图像 (用于SNR计算)
        img_blur = cv2.blur(img_low, (5, 5))
        
        # 生成亮度先验
        img_low_illumination_map = self._illumiantion_map(img_low)
        
        # 加载正常光照图像 (Ground Truth)
        img_gt = cv2.cvtColor(
            cv2.imread(
                os.path.join(
                    self.noraml_img_folder, self.noraml_img_list[index]
                )
            ),
            cv2.COLOR_BGR2RGB,
        )
        
        # 亮度先验归一化: [H, W] → [H, W, 1]
        img_low_illumination_map = np.expand_dims(
            img_low_illumination_map / 255, axis=-1
        )
        
        # 转换事件为PyTorch张量
        event_input_torch = torch.from_numpy(event_input)

        # ==================== 3. 裁剪图像和事件 ====================
        crop_img_list, crop_event_list = self._crop(
            [
                img_low,              # 低光照图像
                img_gt,               # GT图像
                img_low_illumination_map,  # 亮度先验
                img_blur              # 模糊图像
            ],
            [event_input_torch],
        )

        # ==================== 4. 生成Voxel Grid ====================
        # 这是事件特征提取的关键步骤!
        input_voxel_grid_list = []
        for crop_event in crop_event_list:
            # 时间归一化: 从0开始
            crop_event[:, 0] = crop_event[:, 0] - crop_event[0, 0]
            # 生成Voxel Grid
            input_voxel_grid = self._generate_voxel_grid(crop_event)
            input_voxel_grid_list.append(input_voxel_grid)
        
        # 清理内存
        del (
            event_input,
            event_input_torch,
            crop_event_list,
        )

        # ==================== 5. 构建返回样本 ====================
        sample = {
            "lowligt_image": crop_img_list[0]/255,        # 低光照图像 [3, H, W]
            "normalligt_image": crop_img_list[1]/255,     # GT图像 [3, H, W]
            "event_free": input_voxel_grid_list[0],       # Voxel Grid [32, H, W]
            "lowlight_image_blur": crop_img_list[3]/255,  # 模糊图像 [3, H, W]
            "ill_list": [
                crop_img_list[2],                          # 亮度先验 [1, H, W]
            ],
            "seq_name": self.seq_name,
            "frame_id": self.low_img_list[index].split(".")[0],
        }

        # 减少内存占用
        self.events = None

        return sample


def get_eglol_withNE_dataset(
    dataset_root,
    center_cropped_height,
    random_cropped_width,
    is_train,
    is_split_event,
    voxel_grid_channel
):
    """
    获取完整的EvLight数据集
    
    将数据集根目录下所有序列组合成一个ConcatDataset
    
    Args:
        dataset_root: 数据集根目录
        center_cropped_height: 裁剪高度
        random_cropped_width: 裁剪宽度
        is_train: 是否训练模式
        is_split_event: 是否使用分割事件文件
        voxel_grid_channel: Voxel Grid时间通道数
        
    Returns:
        ConcatDataset: 包含所有序列的数据集
    """
    all_seqs = os.listdir(dataset_root)
    all_seqs.sort()

    seq_dataset_list = []

    for seq in all_seqs:
        if os.path.isdir(os.path.join(dataset_root, seq)):
            # 逐个加载每个序列
            seq_dataset_list.append(
                eglol_withNE_dataset(
                    dataset_root,
                    center_cropped_height,
                    random_cropped_width,
                    seq,
                    is_train,
                    voxel_grid_channel,
                    is_split_event
                )
            )
    return ConcatDataset(seq_dataset_list)
