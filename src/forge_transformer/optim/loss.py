import torch
import torch.nn as nn


def cross_entropy_loss(logits, targets):  # logits[b, l, v] 推理结果 targets[b, l] 目标结果
    logits_f32 = logits.to(torch.float32)

    max_logits, _ = torch.max(logits_f32, dim=-1, keepdim=True)  # [b, l, v] -> [n, l, 1]
    shifted = logits_f32 - max_logits  # 广播机制 [b, l, v]
    # 支持任意批维度
    sum_exp = torch.sum(torch.exp(shifted), dim=-1, keepdim=False)  # [b, l] 作为 seq_len 的每一token预测的soft总分
    log_sum_exp = torch.log(sum_exp) + max_logits.squeeze(-1)  # max_logits: [b, l, 1] -> [b, l]

    # targets.unsqueeze(-1) [b, l] -> [b, l, 1]  logits: [b, l, v] take_along_dim ->[b, l, 1]
    true_logits = torch.take_along_dim(logits_f32, targets.unsqueeze(-1), dim=-1).squeeze(-1)  # [b, l]

    tmp = log_sum_exp - true_logits
    return tmp.mean()  # 对所有元素求平均
