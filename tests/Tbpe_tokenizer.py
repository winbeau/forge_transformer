from forge_transformer.bpe import Tokenizer
from forge_transformer import BASE_DIR


tok = Tokenizer.from_files(
    BASE_DIR / "bpe_model/vocab.json",
    BASE_DIR / "bpe_model/merges.txt",
    ["<|endoftext|>"],
)

print("=" * 30 + "Test: bpe tokenizer" + "=" * 30)
print("请输入word:")

while True:
    try:
        line = input()
        for s in line.split():
            ids = tok.encode(s)
            print("ids:", ids)
            print("len:", len(ids))
            print("decoded:", tok.decode(ids))

    except EOFError:
        print("=" * 30 + "Test End" + "=" * 30)
