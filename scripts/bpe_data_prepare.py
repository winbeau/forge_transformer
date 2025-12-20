#!/usr/bin/env python


"""
Usage:
    python scripts/bpe_data_prepare.py
"""

from forge_transformer import DATA_DIR
from forge_transformer.bpe import preview_txt, add_endoftext

from datasets import load_dataset


def main():
    # 加载数据
    data = load_dataset("roneneldan/TinyStories")
    print("=" * 70)
    print("data shape")
    print(data)
    print("=" * 70)

    # 保存数据
    data_path = DATA_DIR / "TinyStories"
    train_file = data_path / "train.txt"
    valid_file = data_path / "valid.txt"
    train_out_file = data_path / "train_with_eot.txt"
    valid_out_file = data_path / "valid_with_eot.txt"
    data_path.mkdir(parents=True, exist_ok=True)
    if train_file.exists():
        print(f"{train_file} exists")
    else:
        with open(train_file, "w", encoding="utf-8") as f:
            for r in data["train"]:
                f.write(r["text"].replace("\n", " ") + "\n")  # 保证一个数据一行文本
            print(f"{train_file} saved. ")

    if valid_file.exists():
        print(f"{valid_file} exists")
    else:
        with open(valid_file, "w", encoding="utf-8") as f:
            for r in data["validation"]:
                f.write(r["text"].replace("\n", " ") + "\n")
            print(f"{valid_file} saved.")

    preview_txt(2, train_file)

    add_endoftext(train_file, train_out_file)
    add_endoftext(valid_file, valid_out_file)

    preview_txt(2, train_out_file)


if __name__ == "__main__":
    main()
