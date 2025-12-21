import sys
import argparse
import torch
from pathlib import Path

# --- 1. 路径修复 ---
# 确保能够导入 src 下的 forge_transformer 包
# 无论是在项目根目录运行 python scripts/train.py 还是在 scripts 目录运行都能生效
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 指向项目根目录 (forge_transformer/)
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# --- 2. 导入组件 ---
from forge_transformer.trainer import TrainingConfig
from forge_transformer.trainer_class import Trainer
from forge_transformer.utils.loader import load_token_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Forge Transformer CLI Training Script")

    # === 数据路径 (必需) ===
    parser.add_argument("--train_data", type=str, required=True, help="Path to training tokens (.npy)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation tokens (.npy)")

    # === 训练超参数 ===
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--total_steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--lr_max", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup_iters", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")

    # === 模型架构 ===
    parser.add_argument("--context_len", type=int, default=256, help="Max sequence length (context window)")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")

    # === 系统与 IO ===
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--ckpt_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every N steps")

    # === 断点续传 ===
    parser.add_argument("--resume_from", type=str, default=None, help="Path to .pt checkpoint to resume from")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading datasets...")
    print(f"  Train: {args.train_data}")
    print(f"  Val:   {args.val_data}")

    # 1. 加载数据 (使用 mmap 模式，内存高效)
    train_data = load_token_dataset(args.train_data)
    val_data = load_token_dataset(args.val_data)

    # 2. 构建配置 (使用命令行参数覆盖默认值)
    # 提示: 如果 TrainingConfig 是 dataclass，可以直接传入 args.__dict__ 的子集，
    # 但为了稳健，这里手动映射核心参数
    cfg = TrainingConfig(
        # 数据
        batch_size=args.batch_size,
        context_len=args.context_len,
        vocab_size=args.vocab_size,
        # 优化
        total_steps=args.total_steps,
        lr_max=args.lr_max,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        # 模型
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        # IO & 系统
        device=args.device,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
        ckpt_every=args.ckpt_every,
        eval_every=args.eval_every,
        ckpt_path_to_resume=args.resume_from,
    )

    print(f"Configuration loaded. Device: {cfg.device}")

    # 3. 初始化并运行训练器
    trainer = Trainer(cfg, train_data, val_data)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Stop] Training interrupted by user.")
    except Exception as e:
        print(f"\n[Error] Training failed: {e}")
        raise e


if __name__ == "__main__":
    main()
