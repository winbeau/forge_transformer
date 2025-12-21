import torch
import torch.nn as nn
from forge_transformer.model import MultiHeadSelfAttention


class SkipRoPE(nn.Module):
    def forward(self, x, positions=None):
        return x


@torch.no_grad()
def main():
    begin = " Test: Attention(Without RoPE) "
    end = " Test End "
    d = (len(begin) - len(end)) // 2
    print("=" * 25 + begin + "=" * 25)

    d_model = 128
    num_heads = 4
    seq_len = 8
    x = torch.randn(2, seq_len, d_model)

    # 1. 官方 MHA (设置 batch_first=True)
    torch_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=False)

    # 2. 你的 MHSA
    my_mha = MultiHeadSelfAttention(d_model, num_heads, seq_len=seq_len)

    # 3. 核心：参数对齐
    # 官方把 QKV 权重拼在一起存在 in_proj_weight (3*d_model, d_model)
    w_q, w_k, w_v = torch_mha.in_proj_weight.chunk(3)
    my_mha.W_Q.weight.copy_(w_q)
    my_mha.W_K.weight.copy_(w_k)
    my_mha.W_V.weight.copy_(w_v)
    my_mha.W_O.weight.copy_(torch_mha.out_proj.weight)

    # 4. 技巧：暂时禁用你的 RoPE 以进行纯逻辑对齐
    # 将 RoPE 替换为直接返回原值的 Identity
    original_rope = my_mha.rope
    setattr(my_mha, "rope", SkipRoPE())

    # 5. 统一掩码
    # 官方 MHA 的 mask 通常是 (L, L) 的 bool 矩阵，True 表示遮蔽
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    # 6. 计算输出
    # 注意：你的 MHSA 内部默认自带了下三角 mask，确保逻辑一致
    out_my = my_mha(x)
    out_torch, _ = torch_mha(x, x, x, attn_mask=mask, need_weights=False)

    # 恢复 RoPE
    setattr(my_mha, "rope", original_rope)

    diff = (out_my - out_torch).abs().max().item()
    print(f"Attention 最大绝对误差: {diff:.2e}")
    # 允许稍大的误差（由于 Softmax 内部处理差异）
    assert diff < 1e-5, "Attention 对齐失败！"
    print("- Attention 逻辑对齐通过")
    print("=" * (25 + d) + end + "=" * (25 + d))


if __name__ == "__main__":
    main()
