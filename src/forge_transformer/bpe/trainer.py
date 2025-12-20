from pathlib import Path
import regex as re
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def get_word_freqs_stream(text_path: str | Path):
    # 使用 ChatGPT2 的正则
    PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    compiled_pat = re.compile(PAT)

    counts = Counter()

    # 逐行流式处理
    with open(text_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=2119489, unit="lines"):
            if not line:
                continue
            words = compiled_pat.findall(line)  # 对当前行进行正则切分

            counts.update(words)  # 更新计数器

    print(f"Unique words: {len(counts)}")

    vocab_splits = {word: [b for b in word.encode("utf-8")] for word in counts}
    """
    vocab_splits:
        {
            "abc": [97, 98, 99],
            "中": [228, 184, 173]
        }
    """

    return counts, vocab_splits


# 统计当前所有相邻 pair 的频率
def get_stats(
    vocab_splits: Dict[str, List[int]],
    word_freqs: Dict[str, int],
) -> Dict[Tuple[int, int], int]:
    counts = defaultdict(int)
    for word, ids in vocab_splits.items():
        freq = word_freqs[word]
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] += freq
    return counts


def merge_vocab(
    pair: Tuple[int, int],
    new_id: int,
    vocab_splits: Dict[str, List[int]],
):
    p0, p1 = pair
    for word in vocab_splits:
        ids = vocab_splits[word]
        if len(ids) < 2:
            continue

        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(new_id)
                i += 2  # 跳过已合并的两个元素
            else:
                new_ids.append(ids[i])
                i += 1
        vocab_splits[word] = new_ids
    return vocab_splits
