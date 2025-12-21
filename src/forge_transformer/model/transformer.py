import torch
import torch.nn as nn

from .layers import Embedding, RMSNorm, Linear
from .block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        max_seq_len=1024,
    ):
        super().__init__()
        self.token_emb = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.out = Linear(d_model, vocab_size)  # 线性层输出

    def forward(self, token_ids):
        x = self.token_emb(token_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 由于 pre-norm 输出前要再进行一次 层归一化
        return self.out(x)
