#!/usr/bin/env python

"""
Usage:
    python tests/check_bpe_encode.py --path ./data/TinyStories/train_tokens.npy
"""

import numpy as np
import argparse


def check_tokens(args):
    file_path = args.path
    target_id = args.id
    print(f"Loading {file_path}...")
    # 使用 mmap_mode='r' 可以秒开大文件，且不占用过多内存
    data = np.load(file_path, mmap_mode="r")

    total_count = data.size
    # 统计等于 target_id 的数量
    special_count = np.sum(data == target_id)

    percentage = (special_count / total_count) * 100 if total_count > 0 else 0

    print("-" * 30)
    print(f"Total tokens:      {total_count:,}")
    print(f"Target ID ({target_id}):    {special_count:,}")
    print(f"Percentage:        {percentage:.4f}%")
    print("-" * 30)

    if special_count > 0:
        # 找前 5 个出现的位置，确认分布是否正常
        indices = np.where(data == target_id)[0]
        print(f"First 5 occurrences at indices: {indices[:5].tolist()}")
    else:
        print(f"Warning: ID {target_id} not found in the file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to .npy file")
    parser.add_argument("--id", type=int, default=256, help="Token ID to search for")
    args = parser.parse_args()

    check_tokens(args)
