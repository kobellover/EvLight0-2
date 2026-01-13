"""
调试 NaN 问题的脚本
"""
import torch
import sys
sys.path.insert(0, '/home/wzy/EvLight0-2')

from egllie.models.egretinex import PreEnhanceModule, EFEFrequencyHighPass, LearnableGuidedFilter

def check_tensor(name, tensor):
    """检查 tensor 是否有 NaN 或 Inf"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    print(f"{name}: shape={tensor.shape}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, nan={has_nan}, inf={has_inf}")
    return has_nan or has_inf

def test_pre_enhance():
    print("=" * 60)
    print("Testing PreEnhanceModule for NaN issues")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 创建模型
    model = PreEnhanceModule(
        voxel_bins=32,
        base_chs=48,
        lgf_channels=1,
        efe_sigma=12.0,
        lgf_radius=2,
        lgf_hidden_dim=32,
        div_eps=1e-3,
        use_efe=True,
        reflectance_hidden_channels=32,
        clamp_output=True
    ).to(device)
    
    # 创建随机输入
    B, H, W = 2, 256, 256
    I_LL = torch.rand(B, 3, H, W, device=device) * 0.3  # 低光图像，值较小
    V = torch.rand(B, 32, H, W, device=device) * 2 - 1  # 事件 voxel
    
    print("\n--- Input ---")
    check_tensor("I_LL", I_LL)
    check_tensor("V", V)
    
    # 逐步测试
    print("\n--- Step by step forward ---")
    
    # Step 1: RGB 光照先验
    P_Y = 0.299 * I_LL[:, 0:1] + 0.587 * I_LL[:, 1:2] + 0.114 * I_LL[:, 2:3]
    P_max = I_LL.max(dim=1, keepdim=True)[0]
    check_tensor("P_Y", P_Y)
    check_tensor("P_max", P_max)
    
    # Step 2: X_rgb
    X_rgb = torch.cat([P_Y, P_max, I_LL], dim=1)
    check_tensor("X_rgb", X_rgb)
    
    # Step 3: RGB Branch
    L_coarse_raw = model.rgb_branch(X_rgb)
    if check_tensor("L_coarse_raw", L_coarse_raw):
        print("  -> NaN in L_coarse_raw!")
    
    L_coarse = torch.clamp(L_coarse_raw, min=0.01, max=2.0)
    check_tensor("L_coarse (clamped)", L_coarse)
    
    # Step 4: Event Branch
    F_e = model.event_branch(V)
    if check_tensor("F_e", F_e):
        print("  -> NaN in F_e!")
    
    # Step 5: EFE
    F_e_hp = model.efe(F_e)
    if check_tensor("F_e_hp (after EFE)", F_e_hp):
        print("  -> NaN in F_e_hp after EFE!")
    
    # Step 6: to_lgf
    F_e_hp_1ch = model.to_lgf(F_e_hp)
    if check_tensor("F_e_hp_1ch", F_e_hp_1ch):
        print("  -> NaN in F_e_hp_1ch!")
    
    # Step 7: LGF
    L_refined_raw = model.lgf(guide=F_e_hp_1ch, src=L_coarse)
    if check_tensor("L_refined_raw (after LGF)", L_refined_raw):
        print("  -> NaN in L_refined_raw after LGF!")
    
    L_refined = torch.clamp(L_refined_raw, min=0.01, max=2.0)
    check_tensor("L_refined (clamped)", L_refined)
    
    # Step 8: ReflectanceInit
    R_pre, R_prior_raw = model.reflectance_init(I_LL, L_refined, return_intermediates=True)
    if check_tensor("R_prior_raw", R_prior_raw):
        print("  -> NaN in R_prior_raw!")
    if check_tensor("R_pre", R_pre):
        print("  -> NaN in R_pre!")
    
    print("\n--- Full forward pass ---")
    try:
        output = model(I_LL, V, return_event_feature=True)
        R_pre_full, F_e_out = output
        check_tensor("R_pre (full)", R_pre_full)
        check_tensor("F_e (full)", F_e_out)
        print("\nFull forward pass: SUCCESS")
    except Exception as e:
        print(f"\nFull forward pass: FAILED with error: {e}")

def test_lgf_separately():
    """单独测试 LGF 模块"""
    print("\n" + "=" * 60)
    print("Testing LearnableGuidedFilter separately")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    lgf = LearnableGuidedFilter(r=2, channels=1, hidden_dim=32).to(device)
    
    # 创建测试输入
    B, H, W = 2, 256, 256
    guide = torch.rand(B, 1, H, W, device=device)
    src = torch.rand(B, 1, H, W, device=device) * 0.5 + 0.1  # 确保正数
    
    print("Input:")
    check_tensor("guide", guide)
    check_tensor("src", src)
    
    print("\nLGF forward:")
    output = lgf(guide, src)
    check_tensor("LGF output", output)

def test_efe_separately():
    """单独测试 EFE 模块"""
    print("\n" + "=" * 60)
    print("Testing EFEFrequencyHighPass separately")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    efe = EFEFrequencyHighPass(base_sigma=12.0).to(device)
    
    # 创建测试输入
    B, C, H, W = 2, 48, 256, 256
    x = torch.rand(B, C, H, W, device=device) * 2 - 1  # 范围 [-1, 1]
    
    print("Input:")
    check_tensor("x", x)
    
    print("\nEFE forward:")
    output = efe(x)
    check_tensor("EFE output", output)

if __name__ == "__main__":
    test_efe_separately()
    test_lgf_separately()
    test_pre_enhance()

