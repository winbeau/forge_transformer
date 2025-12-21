import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

# 从项目内部模块导入组件
from forge_transformer.optim import AdamW, cross_entropy_loss, clip_gradients, cosine_lr_schedule
from forge_transformer.model import TransformerLM

# 假设 get_batch 等函数已在 utils/__init__.py 中导出，如果没有，请修改为 from forge_transformer.utils.sampler import get_batch
from forge_transformer.utils import get_batch, load_checkpoint, save_checkpoint
from .config import TrainingConfig


class Trainer:
    def __init__(self, cfg: TrainingConfig, train_data: np.ndarray, val_data: np.ndarray):
        """
        初始化训练器
        :param cfg: 训练配置对象 (TrainingConfig)
        :param train_data: 训练集 token 数组 (numpy array)
        :param val_data: 验证集 token 数组 (numpy array)
        """
        self.cfg = cfg
        self.train_data = train_data
        self.val_data = val_data
        self.device = cfg.device

        # 1. 准备文件系统
        self._setup_directories()

        # 2. 初始化日志文件
        log_path = os.path.join(self.cfg.log_dir, f"{self.cfg.log_save_prefix}.log")
        self.log_file = open(log_path, "a", encoding="utf-8")

        # 3. 初始化模型与优化器 (包含断点续传逻辑)
        self.model, self.optimizer, self.start_step = self._init_model_and_optimizer()

    def _setup_directories(self):
        """创建必要的日志和 Checkpoint 目录"""
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

    def _init_model_and_optimizer(self) -> Tuple[nn.Module, torch.optim.Optimizer, int]:
        """构建模型、优化器并尝试加载 Checkpoint"""
        print(f"Initializing model on {self.device}...")

        # 定义模型
        model = TransformerLM(
            vocab_size=self.cfg.vocab_size,
            d_model=self.cfg.d_model,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_layers,
            max_seq_len=self.cfg.context_len,
        )
        model.to(self.device)

        # 定义优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.cfg.lr_max,  # 这里的 lr 只是占位，step 中会动态调整
            betas=self.cfg.betas,
            eps=self.cfg.eps,
            weight_decay=self.cfg.weight_decay,
        )

        # 断点续传逻辑
        start_step = 0
        resume_path = self.cfg.ckpt_path_to_resume
        if resume_path and os.path.exists(resume_path):
            print(f"[Trainer] Resuming training from checkpoint: {resume_path}")
            start_step = load_checkpoint(model, optimizer, resume_path)
            # 在日志中记录续传事件
            self.log_file.write(f"=== Resumed from {resume_path} at step {start_step} ===\n")

        return model, optimizer, start_step

    @torch.no_grad()
    def evaluate(self, num_batches: int = 20) -> float:
        """在验证集上评估 loss"""
        self.model.eval()
        losses = []
        for _ in range(num_batches):
            x, y = get_batch(
                self.val_data, batch_size=self.cfg.batch_size, context_len=self.cfg.context_len, device=self.device
            )
            logits = self.model(x)
            loss = cross_entropy_loss(logits, y)
            losses.append(loss.item())

        self.model.train()
        return float(sum(losses) / len(losses))

    def train(self):
        """执行主训练循环"""
        self.log_file.write(f"=== Training started at {time.ctime()} ===\n")
        print(f"Start training loop from step {self.start_step} to {self.cfg.total_steps}...")

        self.model.train()

        for cur_step in range(self.start_step, self.cfg.total_steps):
            t0 = time.time()

            # --- 1. 获取 Batch ---
            x, y = get_batch(
                self.train_data, batch_size=self.cfg.batch_size, context_len=self.cfg.context_len, device=self.device
            )

            # --- 2. 学习率调度 (Cosine Decay with Warmup) ---
            lr = cosine_lr_schedule(
                t=cur_step,
                lr_max=self.cfg.lr_max,
                lr_min=self.cfg.lr_min,
                warmup_iters=self.cfg.warmup_iters,
                cosine_iters=self.cfg.cosine_iters,
            )
            # 更新优化器组中的学习率
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # --- 3. 前向传播与 Loss 计算 ---
            logits = self.model(x)
            loss = cross_entropy_loss(logits, y)

            # --- 4. 反向传播与参数更新 ---
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_gradients(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

            # 计算 step 耗时
            dt = time.time() - t0

            # --- 5. 日志记录 ---
            if cur_step % self.cfg.log_every == 0:
                loss_val = loss.item()
                msg = (
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"step {cur_step:06d} | "
                    f"loss {loss_val:.4f} | "
                    f"lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | "
                    f"time {dt * 1000:.2f}ms"
                )
                print(msg)
                self.log_file.write(msg + "\n")
                self.log_file.flush()

            # --- 6. 验证集评估 ---
            if (cur_step > 0 and cur_step % self.cfg.eval_every == 0) or cur_step == self.cfg.total_steps - 1:
                val_loss = self.evaluate()
                val_msg = f"[VAL] step {cur_step:06d} | val_loss {val_loss:.4f}"
                print(val_msg)
                self.log_file.write(val_msg + "\n")
                self.log_file.flush()

            # --- 7. Checkpoint 保存 ---
            if cur_step > 0 and cur_step % self.cfg.ckpt_every == 0:
                save_path = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.ckpt_save_prefix}_{cur_step}.pt")
                save_checkpoint(self.model, self.optimizer, cur_step, save_path)
                self.log_file.write(f"[CKPT] Saved to {save_path}\n")

        # --- 8. 训练结束 ---
        final_path = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.ckpt_save_prefix}_final.pt")
        save_checkpoint(self.model, self.optimizer, self.cfg.total_steps, final_path)

        finish_msg = f"=== Training finished at {time.ctime()} ==="
        self.log_file.write(finish_msg + "\n")
        self.log_file.close()
        print(finish_msg)
