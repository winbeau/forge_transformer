from forge_transformer.bpe import Tokenizer
from forge_transformer import BASE_DIR

tok = Tokenizer.from_files(BASE_DIR/"bpe_model/vocab.json", BASE_DIR/"bpe_model/merges.txt", ["<|endoftext|>"])

line = open("./data/TinyStories/train_with_eot.txt","r",encoding="utf-8").readline()
ids = tok.encode(line)
roundtrip = tok.decode(ids)

print("ORIG:", repr(line[:200]))
print("RT  :", repr(roundtrip[:200]))

