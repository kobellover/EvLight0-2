"""
EvLight0-2 模型核心文件 - 基于事件相机的低光照图像增强 (改进版)

预提亮流程完全与 MyProjectFrist/pre_enhance_stage1.py 一致:
============================================================
1. RGB 光照先验提取: P_Y (Y通道), P_max (max RGB)
2. Concat: X_rgb = [P_Y, P_max, I_LL]
3. RGB 分支: X_rgb → L_coarse
4. 事件分支: V → F_e
5. EFE 高通: F_e → F_e_hp
6. Learnable Guided Filter: guide=F_e_hp, src=L_coarse → L_refined
7. 反射图初始化: ReflectanceInitFromIllum(I_LL, L_refined) → R_pre

模型架构:
=========
1. PreEnhanceModule: 完整的预提亮模块 (与 MyProjectFrist 一致)
2. ImageEnhanceNet: 图像增强网络 (使用预提亮结果)
3. EgLlie: 主模型
"""

import torch
from torch import nn
import torch.nn.functional as F
from egllie.models.base_block.Trans import Unet_ReFormer
from egllie.models.guided_filter import LearnableGuidedFilter


# ============================================================================
# 工具函数: RGB 转 YCbCr
# ============================================================================

def rgb_to_ycbcr(img):
    """
    RGB 转 YCbCr，只返回 Y 通道
    
    转换公式 (ITU-R BT.601):
        Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        img: RGB 图像 (B, 3, H, W)，值域 [0, 1]
    
    Returns:
        y: Y 通道 (B, 1, H, W)，值域 [0, 1]
    """
    r = img[:, 0:1, :, :]
    g = img[:, 1:2, :, :]
    b = img[:, 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


# ============================================================================
# EFE 频域高通增强模块
# ============================================================================

class EFEFrequencyHighPass(nn.Module):
    """
    频域高通增强模块 (Event Feature Enhancement - Frequency High Pass)
    
    处理流程:
        1. 对输入特征做 2D FFT
        2. fftshift 将零频移到中心
        3. 应用高斯高通掩膜
        4. ifftshift + ifft2 转回空间域
        5. 取实部作为输出
    
    参数说明:
        base_sigma: 基准 sigma 值（在 256x256 分辨率下的 sigma）
        sigma 缩放策略: sigma = base_sigma * min(H, W) / 256
    """
    
    def __init__(self, base_sigma=12.0, eps=1e-8):
        super(EFEFrequencyHighPass, self).__init__()
        self.base_sigma = base_sigma
        self.eps = eps
        self._cached_mask = None
        self._cached_shape = None
    
    def _build_gaussian_highpass_mask(self, H, W, device, dtype):
        """构造高斯高通掩膜"""
        sigma = self.base_sigma * min(H, W) / 256.0
        center_y = H // 2
        center_x = W // 2
        
        y = torch.arange(H, device=device, dtype=dtype) - center_y
        x = torch.arange(W, device=device, dtype=dtype) - center_x
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        d_squared = yy**2 + xx**2
        
        mask = 1.0 - torch.exp(-d_squared / (2.0 * sigma**2 + self.eps))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入事件特征 (B, C, H, W)
        
        Returns:
            x_hp: 高通滤波后的特征 (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device
        input_dtype = x.dtype
        
        # FFT 操作不支持 FP16，需要转换为 FP32
        if input_dtype == torch.float16:
            x = x.float()
        
        if self._cached_mask is None or self._cached_shape != (H, W):
            self._cached_mask = self._build_gaussian_highpass_mask(H, W, device, torch.float32)
            self._cached_shape = (H, W)
        
        mask = self._cached_mask
        if mask.device != device:
            mask = mask.to(device)
            self._cached_mask = mask
        
        # 2D FFT → fftshift → 高通掩膜 → ifftshift → ifft2 → 取实部
        x_fft = torch.fft.fft2(x)
        x_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        x_shift_hp = x_shift * mask
        x_ishift = torch.fft.ifftshift(x_shift_hp, dim=(-2, -1))
        x_ifft = torch.fft.ifft2(x_ishift)
        x_hp = torch.real(x_ifft)
        
        # 转回原来的 dtype
        if input_dtype == torch.float16:
            x_hp = x_hp.half()
        
        return x_hp


# ============================================================================
# RGB 分支网络 (与 MyProjectFrist 完全一致)
# ============================================================================

class RGBBranch(nn.Module):
    """
    RGB 分支卷积网络
    
    结构（固定三层）:
        Conv 1×1: 5 → C
        LeakyReLU
        Conv 5×5: C → C
        LeakyReLU
        Conv 1×1: C → C  (无非线性)
    
    输入: X_rgb = concat([P_Y, P_max, I_LL], dim=1)，shape (B, 5, H, W)
    输出: L_coarse，shape (B, C, H, W)
    """
    
    def __init__(self, out_channels=1):
        """
        Args:
            out_channels: 输出通道数，默认 1（单通道光照图）
        """
        super(RGBBranch, self).__init__()
        C = out_channels
        
        # Conv 1×1: 5 → C
        self.conv1 = nn.Conv2d(5, C, kernel_size=1, stride=1, padding=0, bias=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Conv 5×5: C → C
        self.conv2 = nn.Conv2d(C, C, kernel_size=5, stride=1, padding=2, bias=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Conv 1×1: C → C (无非线性)
        self.conv3 = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=True)
        
        # 初始化：使输出接近输入的亮度先验
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使未训练时输出接近 P_max"""
        # 第一层：提取 P_max (index 1)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.conv1.weight.data[:, 1, :, :] = 1.0  # 取 P_max
        
        # 第二层：恒等映射
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        # 设置中心位置为 1
        self.conv2.weight.data[:, :, 2, 2] = 1.0
        
        # 第三层：恒等映射
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        self.conv3.weight.data[:, :, 0, 0] = 1.0
    
    def forward(self, x):
        """
        Args:
            x: (B, 5, H, W) - concat of [P_Y, P_max, I_LL]
        
        Returns:
            L_coarse: (B, C, H, W) - 粗光照估计
        """
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)  # 第三层不加非线性
        return x


# ============================================================================
# 事件分支网络 (只输出多通道特征)
# ============================================================================

class EventBranch(nn.Module):
    """
    事件分支卷积网络
    
    结构:
        Conv 1×1: BINS → base_chs (多通道特征)
        LeakyReLU
        3×3 DWConv: base_chs → base_chs (depthwise)
        LeakyReLU
        Conv 1×1: base_chs → base_chs (无非线性)
    
    输入: V (事件 voxel grid)，shape (B, BINS, H, W)
    输出: F_e (B, base_chs, H, W) - 多通道事件特征
    """
    
    def __init__(self, in_channels=16, out_channels=48):
        """
        Args:
            in_channels: 输入通道数（voxel grid bins 数）
            out_channels: 输出通道数 (base_chs)
        """
        super(EventBranch, self).__init__()
        C = out_channels
        
        # Conv 1×1: BINS → C
        self.conv1 = nn.Conv2d(in_channels, C, kernel_size=1, stride=1, padding=0, bias=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # 3×3 DWConv: C → C, groups=C, padding=1
        self.dwconv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C, bias=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Conv 1×1: C → C (无非线性)
        self.conv3 = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, BINS, H, W) - 事件 voxel grid
        
        Returns:
            F_e: (B, base_chs, H, W) - 多通道事件特征
        """
        x = self.act1(self.conv1(x))
        x = self.act2(self.dwconv(x))
        F_e = self.conv3(x)  # (B, base_chs, H, W) 多通道特征
        
        return F_e


# ============================================================================
# 反射图初始化模块 (与 MyProjectFrist 完全一致)
# ============================================================================

class ReflectanceInitFromIllum(nn.Module):
    """
    反射图初始化模块 (Reflectance Initialization from Illumination)
    
    功能: 从低光图像和精细光照图生成预提亮反射图 R_pre
    
    处理流程:
        1. R_prior_raw = I_ll / (L_refined + eps)  -- 逐元素除法
        2. R_prior = softmax(R_prior_raw, dim=1)   -- 通道维 softmax
        3. x = concat(I_ll, R_prior)              -- 拼接得到 6 通道
        4. R_pre = ConvStack(x)                   -- CNN 映射到 3 通道
    
    ConvStack 结构（固定三层）:
        Conv 1×1: 6 → hidden_channels, LeakyReLU
        Conv 5×5: hidden_channels → hidden_channels, LeakyReLU
        Conv 1×1: hidden_channels → 3  (无非线性)
    """
    
    def __init__(self, hidden_channels=32, eps=1e-3, clamp_output=True):
        """
        Args:
            hidden_channels: ConvStack 中间层通道数
            eps: 除法时的 epsilon，防止除零 (增大到 1e-3 提高数值稳定性)
            clamp_output: 是否将输出 clamp 到 [0, 1]
        """
        super(ReflectanceInitFromIllum, self).__init__()
        
        self.eps = eps
        self.clamp_output = clamp_output
        
        # ConvStack: 6 → 3 通道
        self.conv1 = nn.Conv2d(6, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.conv3 = nn.Conv2d(hidden_channels, 3, kernel_size=1, stride=1, padding=0, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使未训练时输出接近 R_prior"""
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        for i in range(min(self.conv1.out_channels, 3)):
            self.conv1.weight.data[i, 3 + i, 0, 0] = 1.0
        
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        for i in range(min(self.conv2.out_channels, self.conv2.in_channels)):
            self.conv2.weight.data[i, i, 2, 2] = 1.0
        
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        for i in range(3):
            self.conv3.weight.data[i, i, 0, 0] = 1.0
    
    def forward(self, I_ll, L_refined, return_intermediates=False):
        """
        前向传播
        
        Args:
            I_ll: 低光 RGB 图像 (B, 3, H, W)
            L_refined: 精细光照图 (B, 1, H, W) 或 (B, 3, H, W)
            return_intermediates: 是否返回中间结果
        
        Returns:
            R_pre: 预提亮反射图 (B, 3, H, W)
            (可选) R_prior_raw: 逐元素除法结果 (B, 3, H, W)
        """
        # 广播光照图到 3 通道
        if L_refined.shape[1] == 1:
            L_3ch = L_refined.expand(-1, 3, -1, -1)
        else:
            L_3ch = L_refined
        
        # 数值稳定性: 确保 L_3ch 为正数，避免除法产生极端值
        L_3ch_safe = torch.clamp(L_3ch, min=self.eps)
        
        # R_prior_raw = I_ll / (L_refined + eps)
        R_prior_raw = I_ll / (L_3ch_safe + self.eps)
        
        # 数值稳定性: 限制 R_prior_raw 范围，防止 softmax 溢出
        R_prior_raw_clamped = torch.clamp(R_prior_raw, min=-10.0, max=10.0)
        
        # R_prior = softmax(R_prior_raw, dim=1)
        R_prior = F.softmax(R_prior_raw_clamped, dim=1)
        
        # x = concat(I_ll, R_prior)
        x = torch.cat([I_ll, R_prior], dim=1)
        
        # ConvStack
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        R_pre = self.conv3(x)
        
        if self.clamp_output:
            R_pre = torch.clamp(R_pre, 0.0, 1.0)
        
        if return_intermediates:
            return R_pre, R_prior_raw
        return R_pre


# ============================================================================
# 预提亮模块 (修改版: 多通道做EFE后再转单通道用于LGF)
# ============================================================================

class PreEnhanceModule(nn.Module):
    """
    预提亮模块 (修改版)
    
    完整处理流程:
        1. 提取 RGB 光照先验: P_Y (Y通道), P_max (max RGB)
        2. Concat: X_rgb = [P_Y, P_max, I_LL]
        3. RGB 分支: X_rgb → L_coarse (单通道)
        4. 事件分支: V → F_e (多通道)
        5. EFE 高通: F_e → F_e_hp (多通道)
        6. 转单通道: F_e_hp → F_e_hp_1ch (用于 LGF)
        7. Learnable Guided Filter: guide=F_e_hp_1ch, src=L_coarse → L_refined
        8. 反射图初始化: ReflectanceInitFromIllum(I_LL, L_refined) → R_pre
    
    输出:
        - R_pre: 预提亮图像
        - F_e: 多通道事件特征 (不经过EFE，直接用于后续增强网络)
    """
    
    def __init__(
        self,
        voxel_bins=32,
        base_chs=48,
        lgf_channels=1,
        efe_sigma=12.0,
        lgf_radius=2,
        lgf_hidden_dim=32,
        div_eps=1e-6,
        use_efe=True,
        reflectance_hidden_channels=32,
        clamp_output=True
    ):
        """
        Args:
            voxel_bins: 事件 voxel grid 的通道数（时间 bins）
            base_chs: 事件特征通道数 (用于后续增强网络)
            lgf_channels: LGF 光照引导通道数 (默认 1)
            efe_sigma: EFE 高通滤波的 base_sigma
            lgf_radius: Learnable Guided Filter 的窗口半径
            lgf_hidden_dim: LGF 系数网络隐藏层维度
            div_eps: 除法的 epsilon，防止除零
            use_efe: 是否使用 EFE 高通模块
            reflectance_hidden_channels: 反射图模块隐藏层通道数
            clamp_output: 是否 clamp 输出到 [0, 1]
        """
        super(PreEnhanceModule, self).__init__()
        
        self.base_chs = base_chs
        self.lgf_channels = lgf_channels
        self.div_eps = div_eps
        self.use_efe = use_efe
        
        # RGB 分支 (生成 L_coarse，单通道)
        self.rgb_branch = RGBBranch(out_channels=lgf_channels)
        
        # 事件分支 (生成 F_e 多通道)
        self.event_branch = EventBranch(
            in_channels=voxel_bins, 
            out_channels=base_chs  # 多通道
        )
        
        # EFE 频域高通模块 (对多通道处理)
        self.efe = EFEFrequencyHighPass(base_sigma=efe_sigma)
        
        # 多通道转单通道 (EFE后的特征转为单通道，用于 LGF)
        self.to_lgf = nn.Conv2d(base_chs, lgf_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Learnable Guided Filter (L_coarse 被 F_e_hp_1ch 引导 → L_refined)
        self.lgf = LearnableGuidedFilter(
            r=lgf_radius,
            channels=lgf_channels,  # 单通道
            hidden_dim=lgf_hidden_dim
        )
        
        # 反射图初始化模块 (使用较大的 eps 提高数值稳定性)
        self.reflectance_init = ReflectanceInitFromIllum(
            hidden_channels=reflectance_hidden_channels,
            eps=max(div_eps, 1e-3),  # 确保 eps 至少为 1e-3
            clamp_output=clamp_output
        )

    def forward(self, I_LL, V, return_intermediates=False, return_event_feature=False):
        """
        前向传播
        
        Args:
            I_LL: 低光 RGB 图像 (B, 3, H, W)，值域 [0, 1]
            V: 事件 voxel grid (B, BINS, H, W)
            return_intermediates: 是否返回中间结果
            return_event_feature: 是否返回事件特征 (用于后续增强)
        
        Returns:
            R_pre: 预提亮图像 (B, 3, H, W)
            F_e: 多通道事件特征 (B, base_chs, H, W) - 如果 return_event_feature=True
            intermediates: dict - 如果 return_intermediates=True
        """
        # ============ Step 1: RGB 光照先验提取 ============
        P_Y = rgb_to_ycbcr(I_LL)  # (B, 1, H, W)
        P_max = I_LL.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        
        # ============ Step 2: Concat 三路 ============
        X_rgb = torch.cat([P_Y, P_max, I_LL], dim=1)  # (B, 5, H, W)
        
        # ============ Step 3: RGB 分支 → L_coarse (单通道) ============
        L_coarse_raw = self.rgb_branch(X_rgb)  # (B, 1, H, W)
        # 数值稳定性: 确保 L_coarse 为正数，使用 sigmoid 或 clamp
        L_coarse = torch.clamp(L_coarse_raw, min=0.01, max=2.0)  # 限制在合理范围
        
        # ============ Step 4: 事件分支 → F_e (多通道) ============
        F_e = self.event_branch(V)  # (B, base_chs, H, W)
        
        # ============ Step 5: EFE 频域高通 (对多通道处理) ============
        if self.use_efe:
            F_e_hp = self.efe(F_e)  # (B, base_chs, H, W) 多通道高通
        else:
            F_e_hp = F_e
        
        # ============ Step 6: 多通道转单通道 (用于 LGF) ============
        F_e_hp_1ch = self.to_lgf(F_e_hp)  # (B, 1, H, W) 单通道，用于 LGF
        
        # ============ Step 7: Learnable Guided Filter → L_refined ============
        L_refined_raw = self.lgf(guide=F_e_hp_1ch, src=L_coarse)  # (B, 1, H, W)
        # 数值稳定性: 确保 L_refined 为正数
        L_refined = torch.clamp(L_refined_raw, min=0.01, max=2.0)
        
        # ============ Step 8: 反射图初始化 → R_pre ============
        R_pre, R_prior_raw = self.reflectance_init(I_LL, L_refined, return_intermediates=True)
        
        # 构建返回值
        results = [R_pre]
        
        if return_event_feature:
            # 返回不经过 EFE 的多通道事件特征，用于后续增强
            results.append(F_e)
        
        if return_intermediates:
            intermediates = {
                'P_Y': P_Y,
                'P_max': P_max,
                'L_coarse': L_coarse,
                'F_e': F_e,
                'F_e_hp': F_e_hp,
                'F_e_hp_1ch': F_e_hp_1ch,
                'L_refined': L_refined,
                'R_prior_raw': R_prior_raw,
                'R_pre': R_pre,
            }
            results.append(intermediates)
        
        if len(results) == 1:
            return results[0]
        return tuple(results)


# ============================================================================
# 图像增强网络 (修改版: 复用预提亮模块中的事件特征)
# ============================================================================

class ImageEnhanceNet(nn.Module):
    """
    图像增强网络 (Image Enhancement Network) - 修改版
    
    关键改进: 预提亮模块中提取的 F_e_hp 直接用于后续增强，无需重复提取
    
    处理流程:
        1. PreEnhanceModule: 生成 R_pre (预提亮图像) + F_e_hp (事件特征)
        2. SNR图生成
        3. 图像特征提取 + 特征融合 (使用 F_e_hp)
        4. 主增强网络 (Unet_ReFormer)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.base_chs = cfg.base_chs
        self.snr_factor = float(cfg.snr_factor)

        # 事件-图像特征对齐层
        self.ev_img_align = nn.Conv2d(
            cfg.base_chs * 2,
            cfg.base_chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # ==================== 预提亮模块 (同时输出 R_pre 和 F_e_hp) ====================
        self.pre_enhance = PreEnhanceModule(
            voxel_bins=cfg.voxel_grid_channel,  # 事件 voxel grid 通道数
            base_chs=cfg.base_chs,               # 事件特征通道数 (用于后续增强)
            lgf_channels=1,                      # LGF 光照引导通道数
            efe_sigma=12.0,                      # EFE sigma
            lgf_radius=2,                        # LGF 窗口半径
            lgf_hidden_dim=32,                   # LGF 隐藏层维度
            div_eps=1e-6,                        # 除法 epsilon
            use_efe=True,                        # 使用 EFE
            reflectance_hidden_channels=32,      # 反射图模块隐藏层
            clamp_output=True                    # clamp 到 [0, 1]
        )

        # 注意: 不再需要独立的 ev_extractor，直接使用 PreEnhanceModule 返回的 F_e_hp

        # 图像特征提取器
        self.img_extractor = nn.Sequential(
            nn.Conv2d(3, cfg.base_chs, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # 主增强网络
        self.Unet_ReFormer = Unet_ReFormer(
            dim=cfg.base_chs,
            snr_threshold_list=cfg.snr_threshold_list
        )
    
    def _snr_generate(self, low_img, low_img_blur): 
        """生成信噪比(SNR)图"""
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

        mask = mask * self.snr_factor / (mask_max + 0.0001) 
        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        
        return mask

    def forward(self, batch, return_intermediates=False):
        """
        前向传播
        
        Args:
            batch: 输入数据字典
            return_intermediates: 是否返回中间结果
        
        Returns:
            pred_normal_img: 增强后的图像
            (可选) intermediates: 中间结果字典
        """
        low_light_img = batch["lowligt_image"]
        low_light_img_blur = batch["lowlight_image_blur"]
        event_voxel = batch["event_free"]
        
        # ==================== 预提亮 + 事件特征提取 (一次完成) ====================
        # F_e_hp 直接从预提亮模块获取，无需重复提取
        if return_intermediates:
            enhance_low_img_mid, event_free, pre_intermediates = self.pre_enhance(
                low_light_img, event_voxel, 
                return_intermediates=True, 
                return_event_feature=True
            )
        else:
            enhance_low_img_mid, event_free = self.pre_enhance(
                low_light_img, event_voxel, 
                return_event_feature=True
            )
        
        # 对模糊图像也做预提亮 (用于SNR计算，不需要事件特征)
        enhance_low_img_blur = self.pre_enhance(low_light_img_blur, event_voxel)
        
        # 生成SNR图
        snr_lightup = self._snr_generate(enhance_low_img_mid, enhance_low_img_blur)
        batch['snr_lightup'] = snr_lightup

        snr_enhance = snr_lightup.detach()
        
        # 从预增强图像提取特征
        enhance_low_img = self.img_extractor(enhance_low_img_mid)
        
        # 特征融合 (event_free 来自预提亮模块的 F_e_hp)
        img_event = self.ev_img_align(
            torch.concat((event_free, enhance_low_img), dim=1)
        )
        
        # 主增强网络
        pred_normal_img = self.Unet_ReFormer(
            img_event,
            enhance_low_img_mid,
            enhance_low_img,
            snr_enhance,
            event_free  # F_e_hp，经过 EFE 高通滤波的事件特征
        )

        if return_intermediates:
            intermediates = {
                'L_coarse': pre_intermediates['L_coarse'],
                'L_refined': pre_intermediates['L_refined'],
                'R_prior_raw': pre_intermediates['R_prior_raw'],
                'R_pre': pre_intermediates['R_pre'],
                'F_e_hp': pre_intermediates['F_e_hp'],
            }
            return pred_normal_img, intermediates

        return pred_normal_img


# ============================================================================
# 保留原始 IllumiinationNet (为了兼容性，但不再使用)
# ============================================================================

class IllumiinationNet(nn.Module):
    """
    [已弃用] 光照估计网络 (Illumination Estimation Network)
    
    注意: 新的预提亮流程使用 PreEnhanceModule，此模块仅保留用于兼容性
    """
    
    def __init__(self, cfg):
        super().__init__()

        self.ill_extractor = nn.Sequential(
            nn.Conv2d(
                cfg.illumiantion_level + 3,
                cfg.illumiantion_level * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                cfg.illumiantion_level * 2,
                cfg.base_chs,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.ill_level = cfg.illumiantion_level
        self.illumiantion_set = cfg.illumiantion_set

        self.reduce = nn.Sequential(
            nn.Conv2d(cfg.base_chs, 1, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, batch):
        ill_list = [int(num) for num in self.illumiantion_set]
        inital_ill = torch.cat(
            [batch["ill_list"][i] for i in ill_list], dim=1
        )
        
        pred_illu_feature = self.ill_extractor(
            torch.concat((inital_ill, batch['lowligt_image']), dim=1)
        )
        
        pred_illumaintion = self.reduce(pred_illu_feature)

        return pred_illumaintion, pred_illu_feature


# ============================================================================
# 主模型: EgLlie (EvLight0-2)
# ============================================================================

class EgLlie(nn.Module):
    """
    EvLight0-2 主模型 (Event-guided Low-Light Image Enhancement - 改进版)
    
    预提亮流程与 MyProjectFrist 完全一致:
        1. RGB 光照先验 (P_Y, P_max)
        2. RGB 分支 → L_coarse
        3. 事件分支 → F_e → EFE → F_e_hp
        4. Learnable Guided Filter: L_coarse 被 F_e_hp 引导 → L_refined
        5. ReflectanceInitFromIllum → R_pre
    
    模型结构:
        输入 → ImageEnhanceNet(含PreEnhanceModule) → 输出
    """
    
    def __init__(self, cfg) -> None:
        super().__init__()
        # 保留 IllumiinationNet 用于兼容性 (可选使用)
        self.IllumiinationNet = IllumiinationNet(cfg.IlluNet)
        # 主增强网络 (包含新的 PreEnhanceModule)
        self.ImageEnhanceNet = ImageEnhanceNet(cfg.ImageNet)

    def forward(self, batch, return_intermediates=False):
        """
        前向传播
        
        Args:
            batch: 输入数据字典
            return_intermediates: 是否返回中间结果用于可视化
        
        Returns:
            outputs: 输出字典，包含 pred 和 gt
        """
        # 注意: 不再使用 IllumiinationNet，预提亮完全在 ImageEnhanceNet 中完成
        # 但保留调用以兼容原有数据流 (batch["illumaintion"] 可能被其他地方使用)
        batch["illumaintion"], batch['illu_feature'] = self.IllumiinationNet(batch)
        
        # 图像增强 (使用新的 PreEnhanceModule)
        if return_intermediates:
            output, intermediates = self.ImageEnhanceNet(batch, return_intermediates=True)
        else:
            output = self.ImageEnhanceNet(batch)

        outputs = {
            'pred': output,
            'gt': batch["normalligt_image"],
        }
        
        if return_intermediates:
            outputs['intermediates'] = intermediates
            outputs['low_light_img'] = batch["lowligt_image"]

        return outputs
