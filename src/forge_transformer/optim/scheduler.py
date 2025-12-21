import math


def cosine_lr_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    cosine_iters: int,
) -> float:
    if t < warmup_iters:  # 线性预热
        return lr_max * (t / max(1, warmup_iters))
    if t <= cosine_iters:  # 余弦退火
        progress = (t - warmup_iters) / max(1, (cosine_iters - warmup_iters))
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))  # 上平移 1.0 乘 0.5 归一化
        return lr_min + (lr_max - lr_min) * cosine_term
    return lr_min  # 退火结束 保持最小学习率
