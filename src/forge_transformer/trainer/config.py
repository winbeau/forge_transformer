import os
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # === Data / batching ===
    batch_size: int = 32
    context_len: int = 256
    vocab_size: int = 10000  # tokenizer vocab size

    # === Model architecture ===
    num_layers: int = 12
    num_heads: int = 16
    d_model: int = 1024
    d_ff: int = field(default=0)  # 默认按 8/3*d_model 计算
    rope_theta: float = 10000.0

    # === Training steps ===
    total_steps: int = 10_000
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 1000

    # === Optimizer (AdamW) ===
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    # === Scheduler ===
    warmup_iters: int = 200
    cosine_iters: int = 10000  # after this, lr = lr_min

    # === Misc ===
    device: str = "cuda"
    target_dir = "./TrainLoopFiles"
    ckpt_dir: str = ""
    ckpt_save_prefix: str = "checkpoint"
    log_dir: str = ""
    log_save_prefix: str = "train"
    ckpt_path_to_resume: str = ""

    # === Derived attributes ===
    def __post_init__(self):  # 自动计算 d_ff, (8/3 × d_model, 且为64的倍数)
        if self.d_ff is None:
            multiple = 64
            raw_dff = int((8 / 3) * self.d_model)
            self.d_ff = multiple * ((raw_dff + multiple - 1) // multiple)
        if self.target_dir is None:
            self.target_dir = "."
        if self.ckpt_dir is None:
            self.ckpt_dir = os.path.join(self.target_dir, "checkpoints")
        if self.log_dir is None:
            self.log_dir = os.path.join(self.target_dir, "logs")

    @staticmethod  # 静态成员函数
    def cal_d_ff(d_m) -> int:  # 计算 d_ff
        multiple = 64
        raw_dff = int((8 / 3) * d_m)
        return multiple * ((raw_dff + multiple - 1) // multiple)
