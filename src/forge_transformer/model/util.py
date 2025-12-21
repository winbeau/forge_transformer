import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = x.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)  # 广播机制
