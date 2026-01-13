"""
调试完整模型 NaN 问题
"""
import torch
import sys
sys.path.insert(0, '/home/wzy/EvLight0-2')

from easydict import EasyDict
from egllie.models.egretinex import EgLlie

def check_tensor(name, tensor):
    """检查 tensor 是否有 NaN 或 Inf"""
    if tensor is None:
        print(f"{name}: None")
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    print(f"{name}: shape={tensor.shape}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, nan={has_nan}, inf={has_inf}")
    return has_nan or has_inf

def create_fake_batch(device, B=2, H=256, W=256):
    """创建模拟 batch 数据"""
    batch = {
        'lowligt_image': torch.rand(B, 3, H, W, device=device) * 0.3,  # 低光图像
        'normalligt_image': torch.rand(B, 3, H, W, device=device),  # GT
        'event_free': torch.rand(B, 32, H, W, device=device) * 2 - 1,  # 事件
        'lowlight_image_blur': torch.rand(B, 3, H, W, device=device) * 0.3,  # 模糊图
        'ill_list': [torch.rand(B, 1, H, W, device=device) * 0.3],  # 亮度先验
    }
    return batch

def test_full_model():
    print("=" * 60)
    print("Testing Full EgLlie Model")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 配置
    cfg = EasyDict({
        'IlluNet': EasyDict({
            'illumiantion_level': 1,
            'illumiantion_set': [0],
            'base_chs': 48,
        }),
        'ImageNet': EasyDict({
            'base_chs': 48,
            'voxel_grid_channel': 32,
            'snr_factor': 3.0,
            'snr_threshold_list': [0.6, 0.5, 0.4],
        }),
    })
    
    # 创建模型
    model = EgLlie(cfg).to(device)
    model.eval()
    
    # 创建 batch
    batch = create_fake_batch(device)
    
    print("\n--- Input tensors ---")
    check_tensor("lowligt_image", batch['lowligt_image'])
    check_tensor("event_free", batch['event_free'])
    
    print("\n--- Forward pass ---")
    with torch.no_grad():
        # 测试 IlluminationNet
        illum, illu_feature = model.IllumiinationNet(batch)
        if check_tensor("illumination (IlluNet)", illum):
            print("  -> NaN in IlluNet output!")
        batch["illumaintion"] = illum
        batch['illu_feature'] = illu_feature
        
        # 测试 PreEnhanceModule
        pre_enhance = model.ImageEnhanceNet.pre_enhance
        low_light_img = batch["lowligt_image"]
        event_voxel = batch["event_free"]
        
        R_pre, F_e, intermediates = pre_enhance(
            low_light_img, event_voxel, 
            return_intermediates=True, 
            return_event_feature=True
        )
        
        print("\n--- PreEnhance intermediates ---")
        for key, val in intermediates.items():
            if check_tensor(key, val):
                print(f"  -> NaN in {key}!")
        
        # 测试完整 ImageEnhanceNet
        print("\n--- ImageEnhanceNet ---")
        try:
            output = model.ImageEnhanceNet(batch)
            if check_tensor("ImageEnhanceNet output", output):
                print("  -> NaN in ImageEnhanceNet output!")
        except Exception as e:
            print(f"ImageEnhanceNet failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试完整模型
        print("\n--- Full model ---")
        try:
            outputs = model(batch, return_intermediates=True)
            if check_tensor("pred", outputs['pred']):
                print("  -> NaN in final output!")
            print("\nFull model: SUCCESS")
        except Exception as e:
            print(f"Full model failed: {e}")
            import traceback
            traceback.print_exc()

def test_with_amp():
    """测试 AMP 是否导致 NaN"""
    print("\n" + "=" * 60)
    print("Testing with AMP (Mixed Precision)")
    print("=" * 60)
    
    device = 'cuda'
    
    # 配置
    cfg = EasyDict({
        'IlluNet': EasyDict({
            'illumiantion_level': 1,
            'illumiantion_set': [0],
            'base_chs': 48,
        }),
        'ImageNet': EasyDict({
            'base_chs': 48,
            'voxel_grid_channel': 32,
            'snr_factor': 3.0,
            'snr_threshold_list': [0.6, 0.5, 0.4],
        }),
    })
    
    model = EgLlie(cfg).to(device)
    model.train()
    
    batch = create_fake_batch(device)
    
    # 测试 FP32
    print("\n--- FP32 forward ---")
    with torch.no_grad():
        outputs = model(batch)
        check_tensor("pred (FP32)", outputs['pred'])
    
    # 测试 AMP
    print("\n--- AMP forward ---")
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = model(batch)
            check_tensor("pred (AMP)", outputs['pred'])

def test_training_step():
    """模拟训练步骤"""
    print("\n" + "=" * 60)
    print("Testing Training Step with Loss")
    print("=" * 60)
    
    device = 'cuda'
    
    cfg = EasyDict({
        'IlluNet': EasyDict({
            'illumiantion_level': 1,
            'illumiantion_set': [0],
            'base_chs': 48,
        }),
        'ImageNet': EasyDict({
            'base_chs': 48,
            'voxel_grid_channel': 32,
            'snr_factor': 3.0,
            'snr_threshold_list': [0.6, 0.5, 0.4],
        }),
    })
    
    model = EgLlie(cfg).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    batch = create_fake_batch(device)
    
    print("Training with AMP...")
    for step in range(3):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(batch)
            pred = outputs['pred']
            gt = outputs['gt']
            
            # 简单的 L1 loss
            loss = torch.nn.functional.l1_loss(pred, gt)
        
        print(f"Step {step}: loss={loss.item():.4f}, pred_nan={torch.isnan(pred).any().item()}")
        
        if torch.isnan(loss):
            print("  -> Loss is NaN! Breaking...")
            break
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    print("Training test completed")

if __name__ == "__main__":
    test_full_model()
    test_with_amp()
    test_training_step()
