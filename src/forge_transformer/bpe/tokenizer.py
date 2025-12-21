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

        # reverse map: bytes -> id (for special tokens / merged strings if needed)
        self.vocab_rev = {v: k for k, v in vocab.items()}

        self.PAT = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # rank for merging (a_id, b_id) -> rank
        self.bpe_ranks: dict[tuple[int, int], int] = {pair: i for i, pair in enumerate(merges)}

        # Build merge result table: (a,b) -> new_id
        # With your vocab format, merged token ids start at 256 or 257.
        # In classic BPE training, the i-th merge creates token id = base_vocab_size + i.
        # Here base is 256 (bytes) + 1 for eot? but your vocab shows 256 is eot, merges start producing 257...
        # So new_id = 257 + rank
        self.merge_new_id_base = 257

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # vocab values are strings; encode to bytes
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

    @lru_cache(maxsize=200000)
    def _bpe_ids(self, token_bytes: bytes) -> tuple[int, ...]:
        """
        BPE over initial byte ids (0..255).
        Uses merges defined as pairs of token ids.
        """
        # initial symbols are byte ids
        word: list[int] = list(token_bytes)
        if len(word) <= 1:
            return tuple(word)

        while True:
            pairs = set(zip(word, word[1:]))
            candidates = [p for p in pairs if p in self.bpe_ranks]
            if not candidates:
                break

            # pick best-ranked merge
            bigram = min(candidates, key=self.bpe_ranks.__getitem__)
            rank = self.bpe_ranks[bigram]
            new_id = self.merge_new_id_base + rank

            first, second = bigram
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

        return tuple(word)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in self.PAT.finditer(text):
            token_bytes = match.group(0).encode("utf-8")
            ids.extend(self._bpe_ids(token_bytes))
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_stream = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return byte_stream.decode("utf-8", errors="replace")

