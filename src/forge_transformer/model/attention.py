import torch
import torch.nn as nn


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
