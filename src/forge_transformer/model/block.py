import torch
import torch.nn as nn

from .layers import RMSNorm, SwiGLU
from .attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):  # FFN + MHSA
    def __init__(self, d_model, num_heads, seq_len=1024, rope_theta=10000):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)  # 门控有参数 存两份
        self.attn = MultiHeadSelfAttention(d_model, num_heads, seq_len=seq_len, rope_theta=rope_theta)
        self.ffn = SwiGLU(d_model)

    def forward(self, x):  # 残差连接 + 预归一化
        x = x + self.attn(self.norm1(x))  # 多头注意力
        x = x + self.ffn(self.norm2(x))  # FFN 前馈神经网络
        return x
