#!/usr/bin/env python

"""
Usage:

"""

import torch
from forge_transformer.model import RMSNorm, Linear, MultiHeadSelfAttention


def main():
    # 模拟超参数
    batch_size = 2
    seq_len = 32
    d_model = 128
    num_heads = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    begin = " Test: transformer components "
    end = " Test End "
    d = (len(begin) - len(end)) // 2
    print("=" * 25 + begin + "=" * 25)

    # 构造模拟输入 [Batch, Seq, Model]
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    print(f"输入形状: {x.shape}")

    # 1. 检查 RMSNorm
    norm = RMSNorm(d_model).to(device)
    x_norm = norm(x)
    assert x_norm.shape == x.shape, "RMSNorm 形状错误"
    print("- RMSNorm 检查通过")

    # 2. 检查 Linear
    proj = Linear(d_model, d_model * 4).to(device)
    x_proj = proj(x)
    assert x_proj.shape == (batch_size, seq_len, d_model * 4), "Linear 形状错误"
    print("- Linear 检查通过")

    # 3. 检查 MultiHeadSelfAttention (含 RoPE)
    attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, seq_len=seq_len).to(device)

    out = attn(x)
    assert out.shape == x.shape, "Attention 输出形状错误"
    print("- MultiHeadSelfAttention & RoPE 检查通过")

    print("=" * (25 + d) + end + "=" * (25 + d))


if __name__ == "__main__":
    main()
