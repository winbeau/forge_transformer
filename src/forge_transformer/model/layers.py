import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]  # 返回对应权重的索引行


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model,
        eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps  # 防止除0
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        origin_dtype = x.dtype
        x = x.to(torch.float32)
        tmp = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(tmp + self.eps)  # rsqrt 平方跟倒数
        return (x_normed * self.weight).to(origin_dtype)


class Linear(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        xavier_std = (2 / (d_in + d_out)) ** 0.5  # Xavier 初始化 保持in/out方差稳定性
        nn.init.trunc_normal_(self.weight, mean=0.0, std=xavier_std, a=-3.0, b=3.0)

    def forward(self, x):
        return x @ self.weight.T


class SwiGLU(nn.Module):
    def __init__(self, d_model, mul_of=64):  # multiple_of 作倍数上取整
        super().__init__()
        d_ff = int((8 / 3) * d_model)
        d_ff = mul_of * ((d_ff + mul_of - 1) // mul_of)
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.W3(x))
