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
