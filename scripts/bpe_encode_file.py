#!/usr/bin/env python

import os
from multiprocessing import Pool, cpu_count
import json
from pathlib import Path

from forge_transformer import BASE_DIR
from forge_transformer.bpe import Tokenizer


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
    output_path.mkdir(parents=True, exist_ok=True)

    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    file_size_bytes = input_path.stat().st_size

    print(f"Using {num_workers} processes to encode {input_txt_path}")

    total_tokens = 0
    buffer = []


if __name__ == "__main__"
    main()
