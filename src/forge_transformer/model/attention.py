import torch
import torch.nn as nn

from .util import softmax
from .layers import Linear


class RoPE(nn.Module):
    def __init__(
        self,
        d_k,
        seq_len,
        Theta=10000.0,
        device=None,
    ):
        super().__init__()
        # theta_i = 10000^(-2i / d_k)
        inv_freq = 1.0 / (Theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,k->ik", t, inv_freq)
        self.register_buffer("cos", torch.cos(freqs))  # 存为 buffer 不会被识别为学习参数
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, x, positions):  # x: [..., seq_len, d_k] position: [..., seq_len]
        cos = self.cos[positions].unsqueeze(1)  # 变为 [Batch, 1, Seq_len, d_k/2]
        sin = self.sin[positions].unsqueeze(1)
        """分隔奇偶 -> 求对应值 -> 按维度堆叠 -> 展平"""
        x0, x1 = x[..., ::2], x[..., 1::2]  # 把 x 拆分为偶数序列、奇数序列
        x_rot = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
        return x_rot.flatten(-2)  # 展开最后两维


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)  # Q: [..., seq_len, d_head
    scores = Q @ K.transpose(-2, -1) / (d_k**0.5)  # K: [..., seq_len, d_head] 不能简单转置
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # 由于要做 softmax 掩码要取反
    attn = softmax(scores, dim=-1)  # 最后一个维度做 softmax
    return attn @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        seq_len=1024,
        rope_theta=10000,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be a multiple of num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
        self.rope = RoPE(self.d_head, seq_len, rope_theta)

    def forward(self, x):
        B, T, C = x.shape  # T = seq_len C = d_model
        positions = torch.arange(T, device=x.device)
        """Q @ KT -> [..., seq_len, seq_len] -> scaled_dot -> @ V -> [..., seq_len, d_head]"""
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        Q = self.rope(Q, positions)  # RoPE: (..., seq_len, d_k) (..., seq_len)
        K = self.rope(K, positions)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        out = scaled_dot_product_attention(Q, K, V, mask=mask)  # 因果下三角
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 创建连续副本 concat
        return self.W_O(out)
