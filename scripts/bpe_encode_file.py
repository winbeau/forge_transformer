#!/usr/bin/env python

"""
Usage:
    python scripts/bpe_encode_file.py \
        --input-txt-path ./data/TinyStories/train_with_eot.txt
        --output-tokens-path ./data/TinyStories/train_tokens.npy
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from forge_transformer import BASE_DIR
from forge_transformer.bpe import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-txt-path", type=Path, required=True)
    parser.add_argument("--output-tokens-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=BASE_DIR / "bpe_model/vocab.json")
    parser.add_argument("--merges-path", type=Path, default=BASE_DIR / "bpe_model/merges.txt")
    parser.add_argument("--special-tokens", type=str, default="<|endoftext|>")
    parser.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 4))
    parser.add_argument("--chunk-size", type=int, default=10000)
    return parser.parse_args()


# 每个进程初始化自己 _tokenizer
def init_worker(vocab_path, merges_path, special_tokens):
    global _tokenizer
    _tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)


# 子进程执行的编码函数
def encode_line(line: str):
    global _tokenizer
    if not line.strip():
        return []
    return _tokenizer.encode(line)


def main(
    input_txt_path: str | Path,
    output_tokens_path: str | Path,
    vocab_path: str | Path = BASE_DIR / "bpe_model/vocab.json",
    merges_path: str | Path = BASE_DIR / "bpe_model/merges.txt",
    special_tokens=None,
    num_workers: int = max(1, cpu_count() - 4),
    chunk_size: int = 10000,
):
    input_path = Path(input_txt_path)
    output_path = Path(output_tokens_path)
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)
    output_path = Path(output_tokens_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    file_size_bytes = input_path.stat().st_size

    print(f"Using {num_workers} processes to encode {input_txt_path}")

    total_tokens = 0
    buffer = []

    with open(output_path, "wb") as f_out:
        with tqdm(
            total=file_size_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Tokenizing",
        ) as pbar:
            with (
                Pool(
                    processes=num_workers,
                    initializer=init_worker,
                    initargs=(vocab_path, merges_path, special_tokens),
                ) as pool,
                open(input_path, "r", encoding="utf-8") as f_in,
            ):
                while True:
                    raw_lines = []
                    for _ in range(chunk_size):
                        line = f_in.readline()
                        if not line:  # EOF
                            break
                        raw_lines.append(line)

                    if not raw_lines:
                        break

                    # 更新进度条
                    pbar.n = f_in.tell()
                    pbar.refresh()

                    # 过滤空行
                    lines = [line for line in raw_lines if line.strip()]

                    if not lines:
                        continue

                    for encoded_ids in pool.imap(encode_line, lines, chunksize=256):
                        buffer.extend(encoded_ids)

                    if buffer:
                        arr = np.array(buffer, dtype=np.uint16)
                        f_out.write(arr.tobytes())
                        total_tokens += len(arr)
                        buffer.clear()

                        pbar.set_postfix(tokens=f"{total_tokens:,}")
    print(f"\nDone! Saved {total_tokens:,} tokens to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        input_txt_path=args.input_txt_path,
        output_tokens_path=args.output_tokens_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=args.special_tokens,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
