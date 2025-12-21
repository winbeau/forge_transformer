from forge_transformer.bpe import Tokenizer
from forge_transformer import BASE_DIR

def encode_with_stats(tokenizer, text):
    kept = 0
    dropped = 0
    for m in tokenizer.PAT.finditer(text):
        b = m.group(0).encode("utf-8")
        for t in tokenizer._get_bpe_tokens(b):
            if t in tokenizer.vocab_rev:
                kept += 1
            else:
                dropped += 1
    return kept, dropped

tok = Tokenizer.from_files(...)

with open("./data/TinyStories/train_with_eot.txt","r",encoding="utf-8") as f:
    total_k = total_d = 0
    for _ in range(200):
        line = f.readline()
        k,d = encode_with_stats(tok, line)
        total_k += k; total_d += d

print("kept:", total_k, "dropped:", total_d, "drop%:", total_d/(total_k+total_d+1e-9))

