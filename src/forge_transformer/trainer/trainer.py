import os
import time
import torch
import torch.nn as nn
import numpy as np

from forge_transformer.optim import AdamW, cross_entropy_loss, clip_gradients, cosine_lr_schedule
from forge_transformer.model import TransformerLM
from forge_transformer.utils import get_batch, load_checkpoint, save_checkpoint
from .config import TrainingConfig


@torch.no_grad()
def evaluate_model_loss(
    model: nn.Module,
    data: np.ndarray,
    config: TrainingConfig,
    num_batches: int = 20,
) -> float:
    model.eval()  # 评估模式

    losses = []

    for _ in range(num_batches):
        x, y = get_batch(
            data,
            batch_size=config.batch_size,
            context_len=config.context_len,
            device=config.device,
        )
        logits = model(x)  # (B, T, vocab) 前向传播 forward
        loss = cross_entropy_loss(logits, y)
        losses.append(loss.item())

    model.train()  # 训练模式
    return float(sum(losses) / len(losses))


def create_model_and_optimizer(cfg: TrainingConfig, ckpt_path_to_resume: str = ""):
    device = cfg.device

    model = TransformerLM(  # 定义模型
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.context_len,
    )
    model.to(device)

    optimizer = AdamW(  # 定义优化器
        model.parameters(),
        lr=cfg.lr_max,  # 默认学习率，会被覆盖掉
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    start_step = 0  # 定义断点
    if ckpt_path_to_resume is not None and os.path.exists(ckpt_path_to_resume):
        print(f"[resume] Loading checkpoint from {ckpt_path_to_resume}")
        start_step = load_checkpoint(model, optimizer, ckpt_path_to_resume)

    return model, optimizer, start_step


def train_loop(
    train_data: np.ndarray,
    val_data: np.ndarray,
    cfg: TrainingConfig,
):
    device = cfg.device
    total_steps = cfg.total_steps
    os.makedirs(cfg.target_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg.log_dir, f"{cfg.log_save_prefix}.log")
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"=== Training started at {time.ctime()} ===\n")

    model, optimizer, start_step = create_model_and_optimizer(cfg, cfg.ckpt_path_to_resume)

    for cur_step in range(start_step, total_steps):
        x, y = get_batch(
            train_data,
            batch_size=cfg.batch_size,
            context_len=cfg.context_len,
            device=device,
        )

        lr_t = cosine_lr_schedule(
            t=cur_step,
            lr_max=cfg.lr_max,
            lr_min=cfg.lr_min,
            warmup_iters=cfg.warmup_iters,
            cosine_iters=cfg.cosine_iters,
        )
        for g in optimizer.param_groups:  # 计算学习率并更新
            g["lr"] = lr_t

        logits = model(x)
        loss = cross_entropy_loss(logits, y)  # 计算当前 step 损失

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # 计算梯度
        grad_norm = clip_gradients(model.parameters(), cfg.grad_clip_norm)  # 梯度裁剪

        optimizer.step()  # 更新参数

        if cur_step % cfg.log_every == 0:  # 训练日志
            loss_val = float(loss.item())
            msg = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"step {cur_step:06d} "
                f"loss {loss_val:.4f} "
                f"lr {lr_t:.2e} "
                f"grad_norm {grad_norm:.2f}"
            )
            log_file.write(msg + "\n")
            log_file.flush()

        if (cur_step % cfg.eval_every == 0 and cur_step > 0) or cur_step == total_steps - 1:  # 验证
            with torch.no_grad():
                val_loss = evaluate_model_loss(model, val_data, cfg)
            val_msg = f"[VAL] step {cur_step:06d} val_loss {val_loss:.4f}"
            log_file.write(val_msg + "\n")
            log_file.flush()

        if cur_step % cfg.ckpt_every == 0 and cur_step > 0:  # Checkpoint 保存
            ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.ckpt_save_prefix}_{cur_step}.pt")
            save_checkpoint(model, optimizer, cur_step, ckpt_path)
            log_file.write(f"[CKPT] Saved to {ckpt_path}\n")
            log_file.flush()

    final_ckpt = os.path.join(cfg.ckpt_dir, f"{cfg.ckpt_save_prefix}_final.pt")
    save_checkpoint(model, optimizer, total_steps, final_ckpt)
    log_file.write(f"=== Training finished at {time.ctime()} ===\n")
    log_file.close()

    return model
