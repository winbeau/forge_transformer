import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay=0.0
    ):  # betas 动量超参
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)  # 默认参数
        super().__init__(params, defaults)  # torch.optim.Optimizer 会自创建 self.param_groups

    @torch.no_grad()  # 禁止梯度追踪
    def step(self, closure=None):
        loss = None
        if closure is not None:  # closure 函数重新计算 loss + backward
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:  # 遍历所有组 每一组是一层的超参数
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:  # group["params"] : 需要被优化的参数(权重)
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]  # 提取当前操作参数(权重)进行计算
                if len(state) == 0:  # 初始化动量参数
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                bias_cor1 = 1 - beta1**t
                bias_cor2 = 1 - beta2**t

                step_size = lr * math.sqrt(bias_cor2) / bias_cor1  # 实际步长 | 简化除法计算

                denom = exp_avg_sq.sqrt().add_(eps)  # 归一化计算滑动平均 v_t | denominator 分母

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)  # 超参数 wd: 让所有参数每步都向 0 缩一点
        return loss
