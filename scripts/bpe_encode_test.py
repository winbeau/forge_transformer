#!/usr/bin/env python3
"""
Parallel BPE encode a large text file into a flat binary token stream.

Key optimizations implemented:
1) Chunk-level parallelism (NOT line-level).
2) Split file into chunks at special token boundaries (e.g. <|endoftext|>).
3) Multiprocessing tuning: imap_unordered + ordered writing with a small buffer.

Example:
  python scripts/bpe_encode_test.py \
    --input-txt-path ./data/TinyStories/train_with_eot.txt \
    --output-tokens-path ./data/TinyStories/train_tokens.bin \
    --vocab-path ./bpe_model/vocab.json \
    --merges-path ./bpe_model/merges.txt \
    --special-tokens "<|endoftext|>" \
    --num-workers 12 \
    --chunks-per-worker 16
"""

import os
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

from forge_transformer import BASE_DIR
from forge_transformer.bpe import Tokenizer


# -----------------------------
# 1) Boundary finder (EOT-aligned)
# -----------------------------
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    mini_chunk_size: int = 4096,
) -> List[int]:
    """
    Chunk the file into parts that can be encoded independently.
    Boundaries are moved forward to the next occurrence of split_special_token.
    Returns a sorted unique list of boundaries including 0 and EOF.
    """
    assert isinstance(split_special_token, (bytes, bytearray)), "split_special_token must be bytes"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = max(1, file_size // desired_num_chunks)

    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    boundaries[0] = 0

    # adjust intermediate boundaries (not first/last)
    for bi in range(1, len(boundaries) - 1):
        initial_position = boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":  # EOF
                boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # boundary at token start
                boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    # unique + sorted
    uniq = sorted(set(boundaries))

    # ensure 0 and file_size exist
    if uniq[0] != 0:
        uniq = [0] + uniq
    if uniq[-1] != file_size:
        uniq.append(file_size)

    # remove overlaps (must be strictly increasing)
    cleaned = [uniq[0]]
    for b in uniq[1:]:
        if b > cleaned[-1]:
            cleaned.append(b)
    return cleaned


# -----------------------------
# 2) Multiprocessing worker
# -----------------------------
_TOKENIZER: Tokenizer | None = None
_INPUT_PATH: Path | None = None
_DTYPE: Any = np.uint16


def init_worker(input_path: Path, vocab_path: Path, merges_path: Path, special_tokens: List[str]):
    """
    Each worker loads its own tokenizer once.
    """
    global _TOKENIZER, _INPUT_PATH, _DTYPE
    _INPUT_PATH = Path(input_path)
    _TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    # pick dtype based on vocab size (avoid overflow)
    vocab_size = len(_TOKENIZER.vocab)
    _DTYPE = np.uint16 if vocab_size <= 65535 else np.uint32


def encode_task(task: Tuple[int, int, int]) -> Tuple[int, bytes, int, int]:
    """
    Encode a byte-range [start, end) from the input file.

    Returns:
      (idx, token_bytes, num_tokens, num_bytes_read)
    """
    global _TOKENIZER, _INPUT_PATH, _DTYPE
    assert _TOKENIZER is not None and _INPUT_PATH is not None

    idx, start, end = task
    with _INPUT_PATH.open("rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    # decode safely; boundaries align to ASCII special token so typically safe
    text = raw.decode("utf-8", errors="ignore")

    ids = _TOKENIZER.encode(text)
    arr = np.asarray(ids, dtype=_DTYPE)
    return idx, arr.tobytes(), int(arr.size), int(end - start)


# -----------------------------
# 3) CLI + Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-txt-path", type=Path, required=True)
    p.add_argument("--output-tokens-path", type=Path, required=True)
    p.add_argument("--vocab-path", type=Path, default=BASE_DIR / "bpe_model/vocab.json")
    p.add_argument("--merges-path", type=Path, default=BASE_DIR / "bpe_model/merges.txt")
    p.add_argument("--special-tokens", type=str, default="<|endoftext|>")
    p.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 4))
    p.add_argument("--chunks-per-worker", type=int, default=16)  # controls number of tasks
    p.add_argument("--mini-chunk-size", type=int, default=4096)  # for boundary scanning
    p.add_argument("--maxtasksperchild", type=int, default=200)  # avoid long-run mem bloat
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_txt_path)
    output_path = Path(args.output_tokens_path)
    vocab_path = Path(args.vocab_path)
    merges_path = Path(args.merges_path)

    # output path is a FILE; create its parent directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # special tokens: ensure list[str]
    special_tokens = [s for s in args.special_tokens.split(",") if s]
    if not special_tokens:
        special_tokens = ["<|endoftext|>"]

    # use bytes token for boundary scan (assume utf-8 ASCII)
    split_token_bytes = special_tokens[0].encode("utf-8")

    file_size = input_path.stat().st_size
    num_workers = max(1, int(args.num_workers))

    # recommended: don't go crazy with worker count for Python BPE
    # (you can still override via CLI)
    print(f"Using {num_workers} processes to encode {input_path}")
    print(f"Special tokens: {special_tokens}")
    print(f"Output: {output_path}")

    # 1) compute boundaries
    desired_num_chunks = max(1, num_workers * max(1, int(args.chunks_per_worker)))
    with input_path.open("rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_token_bytes,
            mini_chunk_size=int(args.mini_chunk_size),
        )

    ranges: List[Tuple[int, int, int]] = []
    for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if e > s:
            ranges.append((i, s, e))

    print(f"Chunk boundaries: {len(boundaries)}  |  Tasks: {len(ranges)}")

    # 2) parallel encode
    total_tokens = 0
    next_to_write = 0
    pending: Dict[int, Tuple[bytes, int, int]] = {}  # idx -> (blob, n_tokens, n_bytes)

    with output_path.open("wb") as f_out, tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Tokenizing",
    ) as pbar:
        with Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(input_path, vocab_path, merges_path, special_tokens),
            maxtasksperchild=int(args.maxtasksperchild),
        ) as pool:
            # Unordered for throughput; we re-order on write
            for idx, blob, n_tok, n_bytes in pool.imap_unordered(encode_task, ranges, chunksize=1):
                pending[idx] = (blob, n_tok, n_bytes)

                # write in original order whenever possible
                while next_to_write in pending:
                    w_blob, w_tok, w_bytes = pending.pop(next_to_write)
                    f_out.write(w_blob)
                    total_tokens += w_tok
                    pbar.update(w_bytes)
                    pbar.set_postfix(tokens=f"{total_tokens:,}")
                    next_to_write += 1

    print(f"\nDone! Saved {total_tokens:,} tokens to {output_path}")
    print("Note: output is a flat binary token stream (not a .npy container).")


if __name__ == "__main__":
    main()

