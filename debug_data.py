"""
检查真实数据的数值范围
"""
import torch
import sys
sys.path.insert(0, '/home/wzy/EvLight0-2')

from egllie.datasets.egsdsd import egsdsd_withNE_dataset

def check_tensor(name, tensor):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    print(f"{name}: shape={tensor.shape}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, nan={has_nan}, inf={has_inf}")
    return has_nan or has_inf

def main():
    print("Loading dataset...")
    dataset = egsdsd_withNE_dataset(
        dataset_root='/home/wzy/SDE/SDE_indoor/sde_in_release',
        height=260,
        width=346,
        seq_name='train',
        is_train=True,
        voxel_grid_channel=32,
        is_split_event=True,
        is_indoor=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 检查前几个样本
    for i in [0, 100, 500, 1000]:
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]
        
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                if check_tensor(key, val):
                    print(f"  WARNING: {key} has NaN/Inf!")
            elif isinstance(val, list):
                for j, v in enumerate(val):
                    if isinstance(v, torch.Tensor):
                        if check_tensor(f"{key}[{j}]", v):
                            print(f"  WARNING: {key}[{j}] has NaN/Inf!")

if __name__ == "__main__":
    main()

