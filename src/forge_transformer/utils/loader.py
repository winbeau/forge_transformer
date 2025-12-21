import numpy as np


def load_token_dataset(path: str, mmap: bool = True):
    if mmap:  # 使用内存映射模式，只在访问时加载数据
        arr = np.load(path, mmap_mode="r")
    else:
        arr = np.load(path)
    print(f"[load_token_dataset] Loaded {path}, shape={arr.shape}, dtype={arr.dtype}")
    return arr
