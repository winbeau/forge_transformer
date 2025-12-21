import torch
from forge_transformer.model import RoPE


def main():
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
    main()
