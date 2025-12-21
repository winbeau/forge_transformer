#!/usr/bin/env python

"""
Usage:
    python tests/check_bpe_tokenizer.py
"""

from forge_transformer.bpe import Tokenizer
from forge_transformer import BASE_DIR

tok = Tokenizer.from_files(
    BASE_DIR / "bpe_model/vocab.json",
    BASE_DIR / "bpe_model/merges.txt",
    ["<|endoftext|>"],
)


def check_eot():
    print("Test <|endoftext|>:")
    print(tok.encode("<|endoftext|>"))


def main():
    begin = " Test: bpe tokenizer "
    end = " Test End "
    d = (len(begin) - len(end)) // 2
    print("=" * 30 + begin + "=" * 30)

    check_eot()

    print("请输入单词（ctrl+d 退出）:")

    while True:
        try:
            line = input()
            for s in line.split():
                ids = tok.encode(s)
                print("ids:", ids)
                print("len:", len(ids))
                print("decoded:", tok.decode(ids))

        except EOFError:
            print("=" * (30 + d) + end + "=" * (30 + d))
            break


if __name__ == "__main__":
    main()
