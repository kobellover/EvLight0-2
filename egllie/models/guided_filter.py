"""
同分辨率可训练引导滤波模块 (Same Resolution Guided Filter)

原始文件: guided_filter.py
修改目的: 
    - 原始版本的 FastGuidedFilter 设计用于超分辨率场景（低分辨率处理 + 高分辨率输出）
    - 本文件针对同分辨率场景：引导图和输入图分辨率相同
    - 应用场景：使用事件数据（边缘信息丰富）引导粗略光照图

主要修改:
    1. GuidedFilterSameRes: 基于原始 GuidedFilter，适配同分辨率场景
    2. LearnableGuidedFilter: 可学习版本，用神经网络替代固定的 A = cov/(var+eps) 公式

注意: 引导图（事件数据）的预处理应在调用本模块之前完成
"""

import torch
from torch import nn


# ============================================================================
# BoxFilter: 盒式滤波器 - O(1) 复杂度实现
# 与原始 box_filter.py 完全一致
# ============================================================================

def diff_x(input, r):
    """
    沿 x 方向（高度方向，dim=2）计算差分
    
    原理：对累积和做差分，得到窗口求和
    对于累积和 S，窗口 [i-r, i+r] 的和 = S[i+r] - S[i-r-1]
    
    Args:
        input: (B, C, H, W) 累积和后的张量
        r: 滤波窗口半径
    
    Returns:
        (B, C, H, W) 差分结果
    """
    assert input.dim() == 4

    left   = input[:, :,         r:2*r+1]
    middle = input[:, :, 2*r+1:         ] - input[:, :,           :-2*r-1]
    right  = input[:, :,        -1:     ] - input[:, :, -2*r-1:    -r-1]

    output = torch.cat([left, middle, right], dim=2)
    return output


def diff_y(input, r):
    """
    沿 y 方向（宽度方向，dim=3）计算差分
    """
    assert input.dim() == 4

    left   = input[:, :, :,         r:2*r+1]
    middle = input[:, :, :, 2*r+1:         ] - input[:, :, :,           :-2*r-1]
    right  = input[:, :, :,        -1:     ] - input[:, :, :, -2*r-1:    -r-1]

    output = torch.cat([left, middle, right], dim=3)
    return output


class BoxFilter(nn.Module):
    """
    盒式滤波器 - O(1) 复杂度
    
    计算流程：
        input → cumsum(dim=2) → diff_x → cumsum(dim=3) → diff_y → 窗口求和
    
    输出是窗口内像素的"和"，需要除以 N 得到"均值"
    """
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


# ============================================================================
# 方案1: GuidedFilterSameRes - 同分辨率引导滤波（固定公式）
# 基于原始 GuidedFilter 修改
# ============================================================================

class GuidedFilterSameRes(nn.Module):
    """
    同分辨率引导滤波 - 固定公式版本
    
    与原始 GuidedFilter 的区别:
        - 明确参数命名: guide（引导图）和 src（源图）
        - 注释更详细，便于理解
    
    数学公式:
        a = cov(guide, src) / (var(guide) + eps)
        b = mean(src) - a * mean(guide)
        output = mean(a) * guide + mean(b)
    """
    
    def __init__(self, r=4, eps=0.01):
        """
        Args:
            r: 滤波窗口半径，窗口大小 (2r+1) x (2r+1)
            eps: 正则化参数，控制平滑强度
        """
        super(GuidedFilterSameRes, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, guide, src):
        """
        Args:
            guide: 引导图 (B, C, H, W) - 事件数据（已预处理）
            src:   源图 (B, C, H, W) - 粗略光照图
        
        Returns:
            output: 滤波后图像 (B, C, H, W)
        """
        n_x, c_x, h_x, w_x = guide.size()
        n_y, c_y, h_y, w_y = src.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N: 归一化因子
        N = self.boxfilter(guide.new_ones((1, 1, h_x, w_x)))

        # 均值
        mean_guide = self.boxfilter(guide) / N
        mean_src = self.boxfilter(src) / N
        
        # 协方差和方差
        cov = self.boxfilter(guide * src) / N - mean_guide * mean_src
        var = self.boxfilter(guide * guide) / N - mean_guide * mean_guide

        # 线性系数
        A = cov / (var + self.eps)
        b = mean_src - A * mean_guide

        # 平滑系数
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * guide + mean_b


# ============================================================================
# 方案2: LearnableGuidedFilter - 可学习引导滤波
# 基于原始 ConvGuidedFilter 修改，适配同分辨率场景
# ============================================================================

class LearnableGuidedFilter(nn.Module):
    """
    可学习的引导滤波 - 同分辨率版本
    
    与原始 ConvGuidedFilter 的区别:
        - 移除上采样操作（同分辨率不需要）
        - 支持任意通道数（原始固定为3通道）
        - 盒式滤波使用分组卷积实现
    
    核心改进:
        固定公式: A = cov / (var + eps)
        可学习:   A = CoefNet([cov, var])
    """
    
    def __init__(self, r=2, channels=1, hidden_dim=32):
        """
        Args:
            r: 滤波窗口半径
            channels: 输入通道数
            hidden_dim: 系数网络隐藏层维度
        """
        super(LearnableGuidedFilter, self).__init__()
        self.r = r
        self.channels = channels
        
        # 盒式滤波器（分组卷积实现）
        self.box_filter = nn.Conv2d(
            channels, channels,
            kernel_size=2*r+1,
            padding=r,
            bias=False,
            groups=channels
        )
        nn.init.constant_(self.box_filter.weight, 1.0 / ((2*r+1)**2))
        
        # 可学习的系数网络
        self.coef_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1)
        )

    def forward(self, guide, src):
        """
        Args:
            guide: 引导图 (B, C, H, W) - 事件数据（已预处理）
            src:   源图 (B, C, H, W) - 粗略光照图
        
        Returns:
            output: 滤波后图像 (B, C, H, W)
        """
        # 均值
        mean_guide = self.box_filter(guide)
        mean_src = self.box_filter(src)
        
        # 协方差和方差
        cov = self.box_filter(guide * src) - mean_guide * mean_src
        var = self.box_filter(guide * guide) - mean_guide * mean_guide
        
        # 可学习系数 A
        A = self.coef_net(torch.cat([cov, var], dim=1))
        b = mean_src - A * mean_guide
        
        # 平滑
        mean_A = self.box_filter(A)
        mean_b = self.box_filter(b)
        
        return mean_A * guide + mean_b




