"""
EvLight 注意力模块 - 用于事件特征提取的通道注意力机制

该文件包含:
===========
1. CA_layer: 通道注意力层 (Channel Attention)
   - 学习通道间的依赖关系
   - 用于多模态特征融合时的通道重要性建模

2. eca_layer: 高效通道注意力 (Efficient Channel Attention)
   - 一种轻量级的通道注意力实现
   - 使用1D卷积代替全连接层，减少参数量
   
3. ECAResidualBlock: 带ECA的残差块
   - 在Trans.py的SNR_enhance中用于事件特征的深度提取
   - 同时用于图像特征和事件特征的处理

在EvLight中的作用:
=================
这些模块主要在SNR_enhance中使用，用于:
- 事件特征的深度提取 (ev_extractor分支)
- 图像特征的深度提取 (img_extractor分支)  
- 三路特征融合时的通道注意力 (fea_align中的CA_layer)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class CA_layer(nn.Module):
    """
    通道注意力层 (Channel Attention Layer)
    
    功能:
    =====
    学习通道间的依赖关系，为不同通道分配不同的重要性权重
    
    在EvLight中的应用:
    =================
    用于SNR_enhance的fea_align模块，对融合后的三路特征
    (加权图像特征 + 加权事件特征 + 注意力特征) 进行通道注意力加权
    
    网络结构:
    =========
    输入 → Conv3x3 → 分支1: Conv3x3→LeakyReLU→Conv3x3→Sigmoid (权重)
                   → 分支2: Identity (残差)
         → 权重 × 残差 + 输入
    
    参数:
    =====
    in_ch: 输入通道数
    """
    def __init__(self, in_ch) -> None:
        super().__init__()
        # 第一个卷积: 用于特征变换
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        
        # 注意力权重生成分支
        # 通过压缩-激活-恢复的方式学习通道权重
        self.conv2 = nn.Sequential(
            # 通道压缩: in_ch → in_ch//2
            nn.Conv2d(in_ch, in_ch//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # 通道恢复: in_ch//2 → in_ch
            nn.Conv2d(in_ch//2, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 输出范围[0,1]的注意力权重
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            加权后的特征 [B, C, H, W]
        
        计算流程:
        =========
        1. res = Conv(x) - 特征变换
        2. cross = Sigmoid(Conv(Conv(x))) - 注意力权重
        3. output = x + res * cross - 残差连接 + 注意力加权
        """
        res = self.conv1(x)       # 特征变换
        cross = self.conv2(x)     # 注意力权重 [0, 1]
        res = res * cross         # 注意力加权
        x = x + res               # 残差连接
        return x


class eca_layer(nn.Module):
    """
    高效通道注意力层 (Efficient Channel Attention Layer)
    
    来源: ECA-Net (CVPR 2020)
    
    核心思想:
    =========
    使用1D卷积代替全连接层来建模通道间的局部依赖关系
    相比SE-Net的全连接层，ECA更加轻量且有效
    
    在EvLight中的应用:
    =================
    用于ECAResidualBlock中，对事件特征和图像特征进行通道注意力增强
    
    计算流程:
    =========
    1. GAP: 全局平均池化获取通道描述符 [B, C, H, W] → [B, C, 1, 1]
    2. 1D卷积: 在通道维度上进行局部交互
    3. Sigmoid: 生成通道注意力权重
    4. 加权: 输入 × 权重
    
    参数:
    =====
    channel: 输入通道数
    k_size: 1D卷积核大小，控制考虑的局部通道范围 (默认3)
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.k_size = k_size
        
        # 1D卷积: 在通道维度上进行局部交互
        # groups=channel 表示对每个通道单独处理，考虑相邻k_size个通道
        self.conv = nn.Conv1d(
            channel, channel, kernel_size=k_size, bias=False, groups=channel
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            通道注意力加权后的特征 [B, C, H, W]
        
        详细步骤:
        =========
        1. 全局平均池化: [B, C, H, W] → [B, C, 1, 1]
        2. 展开用于1D卷积: 调整维度以适应Conv1d
        3. 1D卷积: 学习通道间的局部依赖
        4. Sigmoid: 归一化为[0,1]的权重
        5. 广播乘法: 对原始特征进行加权
        """
        b, c, _, _ = x.size()
        
        # ==================== Step 1: 全局平均池化 ====================
        # [B, C, H, W] → [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # ==================== Step 2: 展开用于1D卷积 ====================
        # 使用unfold进行维度调整，使其适合1D卷积操作
        y = nn.functional.unfold(
            y.transpose(-1, -3),  # 维度转置
            kernel_size=(1, self.k_size),
            padding=(0, (self.k_size - 1) // 2),
        )
        
        # ==================== Step 3: 1D卷积学习通道依赖 ====================
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        
        # ==================== Step 4: Sigmoid激活 ====================
        y = self.sigmoid(y)
        
        # ==================== Step 5: 通道加权 ====================
        # 将权重广播到原始特征尺寸并相乘
        x = x * y.expand_as(x)
        
        return x


class ECAResidualBlock(nn.Module):
    """
    带ECA注意力的残差块 (ECA Residual Block)
    
    核心作用:
    =========
    这是SNR_enhance模块中用于事件特征提取的基本单元!
    
    在EvLight的事件特征处理流程中:
    =============================
    SNR_enhance模块中:
    - self.ev_extractor: 使用ECAResidualBlock堆叠处理事件特征
    - self.img_extractor: 使用ECAResidualBlock堆叠处理图像特征
    
    网络结构:
    =========
    输入 → Conv3x3 → InstanceNorm(半通道)+Identity(半通道) → LeakyReLU 
        → Conv3x3 → ECA通道注意力 → + 输入 (残差) → LeakyReLU → 输出
    
    特点:
    =====
    1. 残差连接: 稳定训练，允许梯度直接回传
    2. ECA注意力: 增强重要通道的特征
    3. 部分InstanceNorm: 只对一半通道做归一化，保持另一半通道的原始分布
       这种设计可以保留更多信息
    
    参数:
    =====
    nf: 特征通道数 (number of features)
    """
    def __init__(self, nf):
        super(ECAResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # ECA通道注意力层
        self.eca = eca_layer(nf)
        
        # InstanceNorm: 只对一半通道进行归一化
        # affine=True 表示有可学习的缩放和偏移参数
        self.norm = nn.InstanceNorm2d(nf//2, affine=True)


    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
               在事件特征处理时，这是从voxel grid提取的事件特征
               
        Returns:
            out: 增强后的特征 [B, C, H, W]
        
        计算流程:
        =========
        1. 第一个卷积层
        2. 分割通道: 一半做InstanceNorm，一半保持不变
        3. 拼接 + LeakyReLU
        4. 第二个卷积层
        5. ECA通道注意力
        6. 残差连接
        7. LeakyReLU激活
        """
        residual = x
        
        # ==================== Step 1: 第一个卷积层 ====================
        out = self.conv1(x)
        
        # ==================== Step 2: 部分InstanceNorm ====================
        # 将特征分成两半
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        # 只对前一半做归一化，后一半保持原样
        # 这样可以保留更多原始特征的统计信息
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu(out)

        # ==================== Step 3: 第二个卷积层 ====================
        out = self.conv2(out)
        
        # ==================== Step 4: ECA通道注意力 ====================
        # 学习通道重要性，增强有用通道，抑制无用通道
        # 对于事件特征，这可以突出有信息量的时间通道
        out = self.eca(out)
        
        # ==================== Step 5: 残差连接 ====================
        out += residual
        out = self.relu(out)
        
        return out
