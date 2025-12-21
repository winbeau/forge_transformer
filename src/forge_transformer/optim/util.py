import torch
import math
from typing import Iterable


@torch.no_grad()
def clip_gradients(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
):
    total_norm_sq = 0.0
    for p in params:  # 计算 L2 范数
        if p.grad is not None:
            total_norm_sq += float(torch.sum(p.grad.data.to(torch.float32) ** 2))
    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)

    return total_norm
