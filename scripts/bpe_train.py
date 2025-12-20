#!/usr/bin/env python

import json
from pathlib import Path

from forge_transformer import DATA_DIR, BASE_DIR
from forge_transformer.bpe import get_word_freqs_stream
from forge_transformer.bpe.trainer import get_stats, merge_vocab


def main(input_path: str | Path, vocab_size: int, output_dir: str | Path = BASE_DIR / "bpe_model"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    word_freqs, vocab_splits = get_word_freqs_stream(input_path)

    # 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    merges = []

    print("Starting BPE training loop...")

    while len(vocab) < vocab_size:
        pairs = get_stats(vocab_splits, word_freqs)

        if not pairs:
            break

        best_pair = max(pairs, key=lambda p: pairs[p])
        best_freq = pairs[best_pair]

        new_id = len(vocab)

        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append(best_pair)

        merge_vocab(best_pair, new_id, vocab_splits)

        if len(vocab) % 1000 == 0:
            try:
                merge_str = vocab[new_id].decode("utf-8")
            except:
                merge_str = str(vocab[new_id])
            print(
                f"Step {len(vocab)}/{vocab_size}: merged {best_pair} -> {new_id} ({repr(merge_str)}), freq={best_freq}"
            )
    print("Writing files ...")

    final_vocab = {str(idx): b_val.decode("utf-8", errors="replace") for idx, b_val in vocab.items()}

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(final_vocab, f, ensure_ascii=False, indent=2)

    # 保存 merges.txt
    with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
        f.write("# version: 0.1\n")
        # 每一行记录: "idx1 idx2" (原 ID)
        for p0, p1 in merges:
            f.write(f"{p0} {p1}\n")

    print(f"Training Complete. Files saved to {output_dir}")
    return vocab, merges

    # if __name__ == "main":


main(input_path=(DATA_DIR / "TinyStories/train_with_eot.txt"), vocab_size=10000)
