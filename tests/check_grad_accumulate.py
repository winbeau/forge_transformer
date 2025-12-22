#!/usr/bin/env python

"""
Usage:
    python tests/check_grad_accumulate.py
"""

import copy
import os
import tempfile

import numpy as np
import torch

from forge_transformer.model import TransformerLM
from forge_transformer.optim import AdamW, clip_gradients, cosine_lr_schedule, cross_entropy_loss
from forge_transformer.trainer import trainer as trainer_mod
from forge_transformer.trainer.config import TrainingConfig
from forge_transformer.utils import get_batch


def _build_config(tmp_dir: str) -> TrainingConfig:
    cfg = TrainingConfig(
        batch_size=2,
        context_len=8,
        vocab_size=128,
        num_layers=1,
        num_heads=2,
        d_model=16,
        total_steps=1,
        log_every=1,
        eval_every=9999,
        ckpt_every=9999,
        lr_max=1e-3,
        lr_min=1e-3,
        warmup_iters=0,
        cosine_iters=1,
        weight_decay=0.0,
        grad_clip_norm=1e9,
        grad_accum_steps=2,
        device="cpu",
        ckpt_dir=os.path.join(tmp_dir, "ckpt"),
        log_dir=os.path.join(tmp_dir, "log"),
        log_save_prefix="grad_accum_test",
        ckpt_save_prefix="grad_accum_test",
    )
    cfg.target_dir = tmp_dir
    return cfg


def _make_fixed_batches(data: np.ndarray, cfg: TrainingConfig):
    np.random.seed(1234)
    batch1 = get_batch(data, cfg.batch_size, cfg.context_len, device=cfg.device)
    batch2 = get_batch(data, cfg.batch_size, cfg.context_len, device=cfg.device)
    return [batch1, batch2]


def _build_model(cfg: TrainingConfig) -> TransformerLM:
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        max_seq_len=cfg.context_len,
    )
    model.to(cfg.device)
    return model


def _max_param_diff(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    max_diff = 0.0
    for name, tensor_a in state_a.items():
        tensor_b = state_b[name]
        diff = (tensor_a - tensor_b).abs().max().item()
        if diff > max_diff:
            max_diff = diff
    return max_diff


def main():
    begin = " Test: Grad Accumulation "
    end = " Test End "
    d = (len(begin) - len(end)) // 2
    print("=" * 25 + begin + "=" * 25)

    torch.manual_seed(42)
    data = np.arange(0, 128, dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = _build_config(tmp_dir)
        batches = _make_fixed_batches(data, cfg)

        model = _build_model(cfg)
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr_max,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

        model_expected = _build_model(cfg)
        model_expected.load_state_dict(copy.deepcopy(model.state_dict()))
        optimizer_expected = AdamW(
            model_expected.parameters(),
            lr=cfg.lr_max,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

        lr_t = cosine_lr_schedule(
            t=0,
            lr_max=cfg.lr_max,
            lr_min=cfg.lr_min,
            warmup_iters=cfg.warmup_iters,
            cosine_iters=cfg.cosine_iters,
        )
        for group in optimizer_expected.param_groups:
            group["lr"] = lr_t

        optimizer_expected.zero_grad(set_to_none=True)
        for x, y in batches:
            loss = cross_entropy_loss(model_expected(x), y)
            (loss / cfg.grad_accum_steps).backward()
        clip_gradients(model_expected.parameters(), cfg.grad_clip_norm)
        optimizer_expected.step()

        batch_calls = {"count": 0}

        def fake_get_batch(*_args, **_kwargs):
            idx = batch_calls["count"]
            if idx >= len(batches):
                raise RuntimeError("get_batch called more times than expected")
            batch_calls["count"] += 1
            return batches[idx]

        def fake_eval(*_args, **_kwargs):
            return 0.0

        def fake_create_model_and_optimizer(_cfg: TrainingConfig, _resume_path: str = ""):
            return model, optimizer, 0

        orig_get_batch = trainer_mod.get_batch
        orig_eval = trainer_mod.evaluate_model_loss
        orig_create = trainer_mod.create_model_and_optimizer

        try:
            trainer_mod.get_batch = fake_get_batch
            trainer_mod.evaluate_model_loss = fake_eval
            trainer_mod.create_model_and_optimizer = fake_create_model_and_optimizer
            trainer_mod.train_loop(data, data, cfg)
        finally:
            trainer_mod.get_batch = orig_get_batch
            trainer_mod.evaluate_model_loss = orig_eval
            trainer_mod.create_model_and_optimizer = orig_create

        assert batch_calls["count"] == cfg.grad_accum_steps, "get_batch calls do not match grad_accum_steps"

        max_diff = _max_param_diff(model, model_expected)
        print(f"Max parameter diff: {max_diff:.2e}")
        assert max_diff < 1e-6, "Gradient accumulation mismatch"

    print("- Grad accumulation behavior matches expectation")
    print("=" * (25 + d) + end + "=" * (25 + d))


if __name__ == "__main__":
    main()
