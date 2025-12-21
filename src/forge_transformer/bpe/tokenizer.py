import json
from functools import lru_cache
from os import error
from pathlib import Path
from pandas.core.computation.parsing import token
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_rev = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []

        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab = {int(i): v.encode("utf-8") for i, v in vocab_data.items()}

        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                merges.append((parts[0].encode(), parts[1].encode()))
        return cls(vocab, merges, special_tokens)

    def bpe(self, token: bytes) -> list[bytes]:
        word = [bytes([b]) for b in token]  # 字节串 -> 字节序列
        pairs = set(zip(word, word[1:]))

        if not pairs:
            return word

        while True:
            candidates = pairs.intersection(self.bpe_ranks.keys())

            if not candidates:
                break  # 没有适合相邻字符可以合并时 直接退出

            bigram = min(candidates, key=self.bpe_ranks.__getitem__)

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word

            if len(word) == 1:
                break  # 合并完只剩一个token(最终合成为一个token), 结束

            pairs = set(zip(word, word[1:]))  # 重新生成pairs用于下一轮

        return word

    @lru_cache(maxsize=200000)
    def _get_bpe_tokens(self, token_bytes: bytes) -> tuple[bytes, ...]:
        word = [bytes([b]) for b in token_bytes]
        pairs = set(zip(word, word[1:]))
        if not pairs:
            return tuple(word)

        while True:
            candidates = pairs.intersection(self.bpe_ranks.keys())
            if not candidates:
                break
            bigram = min(candidates, key=self.bpe_ranks.__getitem__)
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = set(zip(word, word[1:]))
        return tuple(word)

    def encode(self, text: str) -> list[int]:
        ids = []
        for match in self.PAT.finditer(text):
            token_bytes = match.group(0).encode("utf-8")

            # 3. 使用带缓存的 BPE 计算
            for t in self._get_bpe_tokens(token_bytes):
                if t in self.vocab_rev:
                    ids.append(self.vocab_rev[t])
        return ids

    """
    def encode(self, text: str) -> list[int]:
        ids = []
        for match in self.PAT.finditer(text):  # 每个match是正则后的一个单词
            token_bytes = match.group(0).encode("utf-8")  # 转bytes序列

            if len(token_bytes) == 1 and token_bytes in self.vocab_rev:
                ids.append(self.vocab_rev[token_bytes])
                continue

            for t in self.bpe(token_bytes):
                if t in self.vocab_rev:
                    ids.append(self.vocab_rev[t])
        return ids
    """

    def decode(self, ids: list[int]) -> str:
        byte_stream = b"".join(
            self.vocab.get(i, b"\xef\xbf\xbd")
            for i in ids  # \xef\xbf\xbd - Unicode 的 ``
        )
        return byte_stream.decode("utf-8", errors="replace")
