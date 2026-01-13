"""
EvLight Transformer 模块 - 事件特征提取与融合的核心组件

该文件包含以下关键模块:
=========================
1. SNR_enhance: SNR引导的事件-图像特征融合模块
   - 核心思想: 在低信噪比区域更多地依赖事件特征，在高信噪比区域更多地依赖图像特征
   - 事件相机在低光照下有天然优势，能捕捉到普通相机无法捕捉的细节
   
2. IG_MSA (Image-Guided Multi-head Self-Attention): 多头自注意力机制
   - 用于捕捉全局特征依赖关系
   
3. IGAB (Image-Guided Attention Block): 注意力块
   - 堆叠多个IG_MSA和FeedForward层
   
4. Unet_ReFormer: 基于Transformer的U-Net主干网络
   - Encoder-Bottleneck-Decoder结构
   - 在每个层级都使用SNR_enhance进行事件特征融合

事件特征在网络中的流动路径:
==========================
event_voxel [B,32,H,W] 
    → ev_extractor (egretinex.py) 
    → event_free [B,48,H,W]
    → 与img_feature拼接后通过ev_img_align融合
    → 进入Unet_ReFormer
    → 在各层级的SNR_enhance中与图像特征进行自适应融合
    → 输出增强后的图像
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from egllie.models.base_block.ScConv_block import ECAResidualBlock,CA_layer


class SNR_enhance(nn.Module):
    """
    SNR引导的事件-图像特征融合模块 (Signal-to-Noise Ratio Guided Enhancement)
    
    核心思想:
    =========
    事件相机的优势在于高动态范围，在低光照/低信噪比区域能够捕捉到更多信息。
    因此，该模块根据SNR图自适应地选择使用图像特征还是事件特征:
    - 高SNR区域 (信号强): 主要使用图像特征 (权重0.7)
    - 低SNR区域 (噪声大): 主要使用事件特征 (权重0.7)
    
    这是EvLight项目中事件特征提取与融合的核心创新点!
    
    网络结构:
    =========
    1. 图像特征分支: ECAResidualBlock × depth
    2. 事件特征分支: ECAResidualBlock × depth  
    3. 基于SNR的加权融合
    4. 通道注意力融合 (CA_layer)
    
    参数:
    =====
    channel: 特征通道数 (默认48或其倍数)
    snr_threshold: SNR阈值，用于划分高/低SNR区域
    depth: 特征提取的深度 (ECAResidualBlock的数量)
    """
    
    def __init__(
        self,
        channel,
        snr_threshold,
        depth
    ):
        super().__init__()
        self.channel = channel
        self.depth = depth
        
        # ==================== 图像特征提取分支 ====================
        # 使用ECAResidualBlock进行图像特征的深度提取
        # ECA (Efficient Channel Attention) 提供通道维度的注意力
        self.img_extractor = nn.ModuleList()
        
        # ==================== 事件特征提取分支 ====================
        # 同样使用ECAResidualBlock对事件特征进行处理
        # 注意: 这是专门为事件特征设计的提取器，是事件特征处理的核心部分
        self.ev_extractor = nn.ModuleList()

        for i in range(self.depth):
                # 构建depth层ECAResidualBlock用于图像特征
                self.img_extractor.append(ECAResidualBlock(self.channel))
                # 构建depth层ECAResidualBlock用于事件特征
                # 这些层专门用于提取和增强事件特征的表达能力
                self.ev_extractor.append(ECAResidualBlock(self.channel))

        # ==================== 多模态特征融合层 ====================
        # 将图像特征、事件特征、注意力特征三路融合
        # 输入: channel*3 (图像 + 事件 + 注意力)
        # 输出: channel
        self.fea_align = nn.Sequential(
            CA_layer(self.channel*3),  # 通道注意力，学习三路特征的重要性
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Conv2d(self.channel*3,self.channel*1,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
        )

        # SNR阈值: 高于此值为高SNR区域，低于此值为低SNR区域
        self.threshold = snr_threshold


    def forward(self, cnn_fea , snr_map, att_fea, event_free):
        """
        前向传播 - 事件特征与图像特征的SNR引导融合
        
        这是EvLight中事件特征融合的核心函数!
        
        Args:
            cnn_fea: 图像CNN特征 [B, C, H, W]
            snr_map: 信噪比图 [B, 1, H, W]，值域[0,1]
            att_fea: 经过注意力处理的特征 [B, C, H, W]
            event_free: 事件特征 [B, C, H, W] - 这是从voxel grid提取的事件特征!
        
        Returns:
            out: 融合后的特征 [B, C, H, W]
        
        融合策略:
        =========
        1. 根据SNR阈值将区域分为高SNR和低SNR
        2. 高SNR区域: 图像特征权重0.7，事件特征权重0.3
           - 图像信号质量好，主要依赖图像
        3. 低SNR区域: 图像特征权重0.3，事件特征权重0.7
           - 图像噪声大，利用事件相机的高动态范围优势
        4. 三路特征(加权图像+加权事件+注意力特征)拼接后融合
        """
        # ==================== Step 1: 生成SNR权重掩码 ====================
        # 根据阈值将SNR图二值化
        # 低SNR区域 (snr <= threshold): 权重设为0.3 (给图像特征较低权重)
        # 高SNR区域 (snr > threshold):  权重设为0.7 (给图像特征较高权重)
        snr_map[snr_map <= self.threshold] = 0.3 
        snr_map[snr_map > self.threshold] = 0.7
        
        # 事件特征使用反向权重: 1 - snr_map
        # 即: 低SNR区域事件权重0.7，高SNR区域事件权重0.3
        snr_reverse_map = 1-snr_map
        
        # 扩展到所有通道 [B, 1, H, W] → [B, C, H, W]
        snr_map_enlarge = snr_map.repeat(1, self.channel,1,1)
        snr_reverse_map_enlarge = snr_reverse_map.repeat(1, self.channel,1,1)

        # ==================== Step 2: 特征深度提取 ====================
        # 使用ECAResidualBlock对图像特征和事件特征分别进行深度处理
        for i in range(self.depth):
                # 图像特征提取: 包含ECA通道注意力
                cnn_fea = self.img_extractor[i](cnn_fea)
                # 事件特征提取: 同样使用ECA增强事件特征
                # 这是事件特征提取的核心步骤！
                event_free = self.ev_extractor[i](event_free)
        
        # ==================== Step 3: SNR引导的特征加权 ====================
        # 高SNR区域: 从图像特征中选择 (权重0.7)
        # 利用element-wise乘法实现空间自适应加权
        out_img = torch.mul(cnn_fea,snr_map_enlarge)

        # 低SNR区域: 从事件特征中选择 (权重0.7)
        # 事件特征在低光照区域有更好的表现！
        out_ev = torch.mul(event_free,snr_reverse_map_enlarge)

        # ==================== Step 4: 多模态特征融合 ====================
        # 将三路特征拼接后通过通道注意力融合
        # - out_img: SNR加权的图像特征
        # - out_ev: SNR加权的事件特征  
        # - att_fea: 注意力处理后的特征
        out = self.fea_align(torch.concat((out_img,out_ev,att_fea),dim=1))

        # 如果depth为0，直接返回注意力特征(不进行事件融合)
        if self.depth==0:
            return att_fea

        return out


class PreNorm(nn.Module):
    """
    预归一化层 (Pre-Normalization)
    
    在执行函数fn之前先进行LayerNorm归一化
    用于Transformer结构中的稳定训练
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    """GELU激活函数"""
    def forward(self, x):
        return F.gelu(x)


class IG_MSA(nn.Module):
    """
    Image-Guided Multi-head Self-Attention (图像引导多头自注意力)
    
    功能:
    =====
    捕捉特征图中的长程依赖关系，使网络能够关注全局信息
    
    注意力计算:
    ==========
    A = softmax(K^T * Q) * scale
    out = A * V
    
    此外还加入了位置编码(pos_emb)来保留空间位置信息
    
    参数:
    =====
    dim: 特征维度
    dim_head: 每个注意力头的维度 (默认64)
    heads: 注意力头数量 (默认8)
    """
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        
        # Q, K, V 线性投影
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        
        # 可学习的缩放参数
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        
        # 输出投影
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        
        # 位置编码: 使用深度可分离卷积实现
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        前向传播
        
        Args:
            x_in: 输入特征 [B, H, W, C]
        
        Returns:
            out: 输出特征 [B, H, W, C]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        
        # 计算Q, K, V
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        
        # 重排为多头格式: [B, heads, N, dim_head]
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp,),
        )
        v = v 
        
        # 转置用于注意力计算
        # q: [B, heads, dim_head, N]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        
        # L2归一化
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        
        # 注意力计算: A = K^T * Q
        attn = k @ q.transpose(-2, -1)
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        
        # 加权聚合: out = A * V
        x = attn @ v  # [B, heads, dim_head, N]
        x = x.permute(0, 3, 1, 2)  # [B, N, heads, dim_head]
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        
        # 输出投影
        out_c = self.proj(x).view(b, h, w, c)
        
        # 位置编码
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        
        # 内容注意力 + 位置编码
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    """
    前馈网络 (Feed-Forward Network)
    
    Transformer中的FFN层，使用1x1卷积 + 深度可分离卷积实现
    
    结构: Linear(dim→dim*mult) → GELU → DWConv → GELU → Linear(dim*mult→dim)
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),  # 扩展通道
            GELU(),
            nn.Conv2d(
                dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult
            ),  # 深度可分离卷积
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),  # 压缩通道
        )

    def forward(self, x):
        """
        Args:
            x: [B, H, W, C]
        Returns:
            out: [B, H, W, C]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    """
    Image-Guided Attention Block (图像引导注意力块)
    
    堆叠多个IG_MSA和FeedForward层，形成Transformer块
    用于处理融合后的事件-图像特征
    
    结构:
    =====
    for each block:
        x = MSA(x) + x  (残差连接)
        x = FFN(x) + x  (残差连接)
    """
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        for attn, ff in self.blocks:
            x = attn(x) + x  # 自注意力 + 残差
            x = ff(x) + x    # FFN + 残差
        out = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return out


class Unet_ReFormer(nn.Module):
    """
    基于Transformer的U-Net主干网络 (Unet-ReFormer)
    
    这是EvLight中进行事件-图像特征融合和图像增强的主网络!
    
    网络结构:
    =========
    Encoder (下采样):
        每层包含: IGAB → 下采样卷积
        同时对事件特征和SNR图进行对应的下采样
        
    Bottleneck:
        SNR_enhance (事件特征融合) → IGAB
        
    Decoder (上采样):
        每层包含: 上采样 → skip connection融合 → SNR_enhance → IGAB
        
    关键设计:
    =========
    1. 多尺度事件特征融合: 在encoder每层保存事件特征，decoder对应层使用
    2. SNR引导: 使用SNR图在不同尺度上引导事件-图像特征的融合比例
    3. 残差学习: 最终输出 = 网络预测 + 预增强图像
    
    参数:
    =====
    in_dim: 输入维度 (默认3)
    out_dim: 输出维度 (默认3)
    dim: 基础通道数 (默认31)
    level: U-Net层数 (默认2)
    num_blocks: 每层IGAB的block数量 (默认[2,4,4])
    snr_depth_list: 每层SNR_enhance的depth (默认[2,4,6])
    snr_threshold_list: 每层SNR阈值 (默认[0.5,0.5,0.5])
    """
    def __init__(
        self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4], snr_depth_list=[2,4,6], snr_threshold_list=[0.5,0.5,0.5]
    ):
        super(Unet_ReFormer, self).__init__()
        self.dim = dim
        self.level = level
        self.snr_threshold_list = snr_threshold_list
        self.snr_depth_list = snr_depth_list

        # ==================== Encoder 编码器 ====================
        # 逐层构建encoder，每层通道数翻倍
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        # IGAB: Transformer注意力块
                        IGAB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                        ),
                        # 特征下采样: 步长2的4x4卷积
                        nn.Conv2d(
                            dim_level, dim_level * 2, 4, 2, 1, bias=False
                        ),
                        # 图像特征下采样
                        nn.Conv2d(
                            dim_level, dim_level * 2, 4, 2, 1, bias=False
                        ),
                        # 事件特征下采样 - 保持事件特征的多尺度表示!
                        nn.Conv2d(
                            dim_level, dim_level * 2, 4, 2, 1, bias=False
                        ),
                    ]
                )
            )
            dim_level *= 2

        # ==================== Bottleneck 瓶颈层 ====================
        # 在最深层进行事件-图像特征的深度融合
        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )
        # Bottleneck的SNR引导融合 - 在最深层融合事件和图像特征
        self.bottleneck_SNR = SNR_enhance(dim_level,snr_threshold_list[-1],snr_depth_list[-1])

        # ==================== Decoder 解码器 ====================
        # 逐层上采样，每层通道数减半
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        # 转置卷积上采样
                        nn.ConvTranspose2d(
                            dim_level,
                            dim_level // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        # Skip connection融合
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        # IGAB注意力块
                        IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                        # SNR引导的事件特征融合 - 在decoder每层都进行融合!
                        SNR_enhance(dim_level // 2,snr_threshold_list[level - 1 - i],snr_depth_list[level-1-i])
                    ]
                )
            )
            dim_level //= 2

        # ==================== 输出映射层 ====================
        # 将特征映射回RGB图像
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # SNR图下采样用的平均池化
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        

    def forward(self, event_img, enhance_low_img, fea_img, SNR, event_free):
        """
        前向传播 - 事件引导的图像增强主流程
        
        这是EvLight模型的核心处理流程!
        
        Args:
            event_img: 事件-图像融合特征 [B, C, H, W]
                       来自ev_img_align的输出
            enhance_low_img: 预增强后的低光照图像 [B, 3, H, W]
                             用于最后的残差连接
            fea_img: 图像CNN特征 [B, C, H, W]
                     用于SNR_enhance中的图像分支
            SNR: 信噪比图 [B, 1, H, W]
                 用于引导事件-图像特征的融合比例
            event_free: 事件特征 [B, C, H, W]
                        从voxel grid提取的纯事件特征，这是事件特征的载体!
        
        Returns:
            out: 增强后的正常光照图像 [B, 3, H, W]
        
        处理流程:
        =========
        1. Encoder: 逐层下采样，保存各层特征用于skip connection
           - 同时对事件特征、图像特征、SNR图进行下采样
        2. Bottleneck: 深度融合事件和图像特征
        3. Decoder: 逐层上采样，使用skip connection和SNR_enhance
           - 在每层使用对应尺度的事件特征进行融合
        4. 残差连接: output = mapping(features) + enhance_low_img
        """
        
        # 初始化特征
        fea_event_img = event_img  # 事件-图像融合特征
        fea_img = fea_img          # 纯图像特征

        # ==================== Encoder 编码阶段 ====================
        # 存储每层的特征，用于decoder的skip connection
        fea_event_img_encoder = []  # 融合特征
        SNRDownsample_list = []      # 各尺度SNR图
        fea_img_list = []            # 各尺度图像特征
        event_free_list = []         # 各尺度事件特征 (关键!)
        
        for IGAB, FeaDownSample, ImgDownSample, EvDownSample in self.encoder_layers:
            # 通过IGAB进行特征处理
            fea_event_img = IGAB(fea_event_img)  # [B, C, H, W]
            
            # 保存当前尺度的特征
            fea_event_img_encoder.append(fea_event_img)
            event_free_list.append(event_free)    # 保存事件特征!
            SNRDownsample_list.append(SNR)
            fea_img_list.append(fea_img)
            
            # SNR图下采样
            SNR = self.avg_pool(SNR)
            
            # 融合特征下采样
            fea_event_img = FeaDownSample(fea_event_img)
            # 图像特征下采样
            fea_img = ImgDownSample(fea_img)
            # 事件特征下采样 - 保持多尺度事件特征!
            event_free = EvDownSample(event_free)

        # ==================== Bottleneck 瓶颈层 ====================
        # 在最深层进行事件-图像特征的深度融合
        # 这里是事件特征融合的关键位置之一!
        fea_event_img = self.bottleneck_SNR(fea_img, SNR, fea_event_img, event_free)
        fea_event_img = self.bottleneck(fea_event_img)
        
        # ==================== Decoder 解码阶段 ====================
        for i, (FeaUpSample, Fusion, REIGAB, RESNR_enhance) in enumerate(
            self.decoder_layers
        ):
            # 上采样
            fea_event_img = FeaUpSample(fea_event_img)
            
            # Skip connection: 与encoder对应层特征融合
            fea_event_img = Fusion(
                torch.cat([fea_event_img, fea_event_img_encoder[self.level - 1 - i]], dim=1)
            )
            
            # 获取对应尺度的SNR图、图像特征、事件特征
            SNR = SNRDownsample_list[self.level - 1 - i]
            fea_img = fea_img_list[self.level - 1 - i]
            event_free = event_free_list[self.level - 1 - i]  # 对应尺度的事件特征!
            
            # SNR引导的事件特征融合 - decoder每层都进行融合!
            # 这是事件特征在网络中发挥作用的核心位置
            fea_event_img = RESNR_enhance(fea_img, SNR, fea_event_img, event_free)
            
            # IGAB注意力处理
            fea_event_img = REIGAB(fea_event_img)
            

        # ==================== 输出映射 ====================
        # 残差学习: 输出 = 网络预测 + 预增强图像
        # 网络只需学习残差，降低学习难度
        out = self.mapping(fea_event_img) + enhance_low_img    

        return out
