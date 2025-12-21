import torch
import torch.nn as nn
from forge_transformer.model import RoPE, MultiHeadSelfAttention


class SkipRoPE(nn.Module):
    def forward(self, x, positions=None):
        return x


@torch.no_grad()
def align_attention():
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


def check_rope_property():
    begin = " Test: RoPE "
    end = " Test End "
    d = (len(begin) - len(end)) // 2
    print("=" * 30 + begin + "=" * 30)

    d_k = 64
    seq_len = 128
    theta = 10000.0
    rope = RoPE(d_k, seq_len, Theta=theta)

    # 1. 构造两个随机向量 q 和 k
    q = torch.randn(1, 1, 1, d_k)  # [Batch, Head, Seq, Dim]
    k = torch.randn(1, 1, 1, d_k)

    # 2. 计算在位置 m 和 n 时的旋转结果
    m, n = 10, 15

    # 将 q 放在位置 m，k 放在位置 n
    q_m = rope(q, torch.tensor([m]))
    k_n = rope(k, torch.tensor([n]))

    # 计算点积得分
    score1 = (q_m @ k_n.transpose(-1, -2)).item()

    # 3. 将位置整体平移 offset，比如都向后移动 50 个位置
    offset = 50
    m_new, n_new = m + offset, n + offset

    q_m_shifted = rope(q, torch.tensor([m_new]))
    k_n_shifted = rope(k, torch.tensor([n_new]))

    # 计算平移后的点积得分
    score2 = (q_m_shifted @ k_n_shifted.transpose(-1, -2)).item()

    # 4. 验证 score1 是否等于 score2
    diff = abs(score1 - score2)
    print(f"位置 ({m}, {n}) 的得分: {score1:.6f}")
    print(f"平移到 ({m_new}, {n_new}) 的得分: {score2:.6f}")
    print(f"绝对误差: {diff:.2e}")

    assert diff < 1e-5, "RoPE 相对位置特性验证失败！点积随绝对位置发生了变化。"

    print("- RoPE 验证通过")
    print("=" * (30 + d) + end + "=" * (30 + d))


if __name__ == "__main__":
    align_attention()
    check_rope_property()
