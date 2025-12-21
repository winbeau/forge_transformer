import json
from functools import lru_cache
from pathlib import Path
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.special_tokens = special_tokens or []

        self.vocab_rev = {v: k for k, v in vocab.items()}

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pat = re.compile(PAT)

        # merging (a_id, b_id) -> rank | 排名越靠前 频率越高
        # merge.txt 首对token 在bpe_train时分配为257
        self.bpe_ranks: dict[tuple[int, int], int] = {pair: i for i, pair in enumerate(merges)}

        self.merge_new_id_base = 257  # base + rank 快速encode

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # vocab.json key->valude | (97 -> 'a')
        vocab = {int(i): v.encode("utf-8") for i, v in vocab_data.items()}

        merges: list[tuple[int, int]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((int(a), int(b)))

        return cls(vocab, merges, special_tokens)

    @lru_cache(maxsize=200000)  # 历史答案剪枝
    def bpe(self, token_bytes: bytes) -> tuple[int, ...]:  # lru_cache 要求静态数组
        word: list[int] = list(token_bytes)
        if len(word) <= 1:
            return tuple(word)

        while True:
            pairs = set(zip(word, word[1:]))
            candidates = [p for p in pairs if p in self.bpe_ranks]  # 查找所有merge pair
            if not candidates:
                break

            # dict.__getitem__(key) 为python内置方法 <=> dict[key]
            bigram = min(candidates, key=self.bpe_ranks.__getitem__)  # 找最靠前(即最高频)合并
            rank = self.bpe_ranks[bigram]
            new_id = self.merge_new_id_base + rank

            first, second = bigram  # dict[tuple(first, second)] = merge_rank
            new_word: list[int] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break

        return tuple(word)  # 动态list -> 静态tuple

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in self.compiled_pat.finditer(text):
            token_bytes = match.group(0).encode("utf-8")
            ids.extend(self.bpe(token_bytes))
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_stream = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return byte_stream.decode("utf-8", errors="replace")
