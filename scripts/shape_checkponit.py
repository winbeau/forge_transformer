from forge_transformer import BASE_DIR
from pathlib import Path
import torch
import os

def inspect_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: File {checkpoint_path} not found.")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 使用 map_location='cpu' 避免占用 GPU，weights_only=False 以加载包含元数据的字典
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print("-" * 50)
    
    # 1. 检查 Checkpoint 的结构
    if isinstance(checkpoint, dict):
        print(f"Checkpoint structure: Dictionary")
        print(f"Keys found: {list(checkpoint.keys())}")
        
        # 修改加载逻辑部分
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']  # 你的模型权重在这里
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        print(f"Checkpoint structure: Direct State Dict (or other object)")
        state_dict = checkpoint

    print("-" * 50)

    # 2. 遍历参数层级
    total_params = 0
    print(f"{'Layer Name':<60} | {'Shape':<20} | {'Dtype':<10}")
    print("-" * 95)
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            shape_str = str(list(value.shape))
            print(f"{key:<60} | {shape_str:<20} | {str(value.dtype):<10}")
            total_params += value.numel()
        else:
            print(f"Non-tensor key: {key} (Type: {type(value)})")

    print("-" * 95)
    
    # 3. 统计汇总
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # 检查是否有元数据
    for meta_key in ['epoch', 'step', 'loss', 'optimizer', 'scheduler']:
        if isinstance(checkpoint, dict) and meta_key in checkpoint:
            if meta_key == 'optimizer':
                print(f"Found Optimizer state: Yes")
            else:
                print(f"Found {meta_key}: {checkpoint[meta_key]}")

if __name__ == "__main__":
    # 替换为你实际的 .pt 文件路径
    ckpt_path = BASE_DIR / "checkpoints" / "checkpoint_12000.pt" 
    inspect_checkpoint(ckpt_path)
