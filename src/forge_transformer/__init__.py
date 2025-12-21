import torch
from .utils import *

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "CHECKPOINT_DIR",
    "DEVICE",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
