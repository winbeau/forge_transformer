#!/usr/bin/env python3

"""
Usage:
    python scripts/bpe_encode_chunks.py \
        --input ./data/TinyStories/train_with_eot.txt \
        --output ./data/TinyStories/train_tokens.npy
"""

import os
import argparse
from pathlib import Path
from typing import BinaryIO, Any

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from forge_transformer import BASE_DIR
from forge_transformer.bpe import Tokenizer


def find_chunk_boundaries(file: BinaryIO, num_chunks: int, tok: bytes, include_newline: bool = True) -> list[int]:
    file.seek(0, os.SEEK_END)
    size = file.tell()
    if size == 0 or num_chunks <= 1:
        return [0, size]

    chunk_size = size // num_chunks
    boundaries = [0]
    for i in range(1, num_chunks):
        pos = i * chunk_size
        file.seek(max(0, pos - len(tok)))
        chunk = file.read(8192)  # 足够覆盖 token + 换行
        idx = chunk.find(tok)
        if idx != -1:
            new_pos = file.tell() - len(chunk) + idx + len(tok)
            if include_newline:
                file.seek(new_pos)
                nxt = file.read(2)
                if nxt.startswith(b"\r\n"):
                    new_pos += 2
                elif nxt.startswith(b"\n"):
                    new_pos += 1
            boundaries.append(min(new_pos, size))

    boundaries.append(size)
    return sorted(list(set(boundaries)))


def init_worker(v, m, s):
    global _tokenizer
    _tokenizer = Tokenizer.from_files(v, m, s)


def worker_job(args) -> tuple[int, np.ndarray, int]:
    idx, path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)

    text = data.decode("utf-8", errors="ignore")
    ids = []
    for line in text.splitlines(True):
        if line.strip():
            ids.extend(_tokenizer.encode(line))

    arr = np.array(ids, dtype=np.uint16)
    return idx, arr, (end - start)


def main(args):
    input_path = args.input
    output_path = args.output
    num_workers = args.num_workers
    num_chunks = args.desired_chunks if args.desired_chunks > 0 else num_workers * 4
    spec_tok = ["<|endoftext|>"]

    with open(input_path, "rb") as f:
        bounds = find_chunk_boundaries(f, num_chunks, spec_tok[0].encode())

    # pool.imap_unordered(worker_job, spans) 会把 spans 中每个元素传入worker_job函数
    spans = [(i, str(input_path), bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    results: list[Any] = [None] * len(spans)
    total_bytes = input_path.stat().st_size

    print(f"Processing {len(spans)} spans using {num_workers} workers...")

    with Pool(
        num_workers,
        init_worker,
        (str(BASE_DIR / "bpe_model/vocab.json"), str(BASE_DIR / "bpe_model/merges.txt"), spec_tok),
    ) as pool:
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Encoding",
        ) as pbar:
            for idx, arr, s_bytes in pool.imap_unordered(worker_job, spans):
                results[idx] = arr
                pbar.update(s_bytes)

    print("Consolidating tokens...")
    total_tokens = sum(len(a) for a in results)
    final_arr = np.empty(total_tokens, dtype=np.uint16)

    curr = 0
    for arr in results:
        final_arr[curr : curr + len(arr)] = arr
        curr += len(arr)

    np.save(output_path, final_arr)
    print(f"Done. Total tokens: {total_tokens:,}. Saved to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=min(12, max(1, cpu_count() - 4)))
    p.add_argument("--desired-chunks", type=int, default=1024)
    args = p.parse_args()

    main(args)
