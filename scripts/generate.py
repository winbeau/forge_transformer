import torch
import torch.nn.functional as F
import argparse
import sys
import os
from pathlib import Path

# Add src to path so we can import forge_transformer
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from forge_transformer.model.transformer import TransformerLM
from forge_transformer.bpe.tokenizer import Tokenizer

def infer_model_config(state_dict):
    """
    Infers model configuration from the state dictionary.
    """
    # 1. Vocab Size and d_model from token embedding
    if 'token_emb.weight' in state_dict:
        vocab_size, d_model = state_dict['token_emb.weight'].shape
    else:
        raise ValueError("Could not find 'token_emb.weight' in checkpoint.")

    # 2. Number of layers
    # Find the maximum block index
    max_block_idx = -1
    for key in state_dict.keys():
        if key.startswith('blocks.'):
            parts = key.split('.')
            if parts[1].isdigit():
                idx = int(parts[1])
                if idx > max_block_idx:
                    max_block_idx = idx
    num_layers = max_block_idx + 1

    # 3. Number of heads
    # Infer from RoPE or Attention weights
    # RoPE cos shape is [seq_len, head_dim/2]
    # We found blocks.0.attn.rope.cos
    rope_key = 'blocks.0.attn.rope.cos'
    max_seq_len = 1024 # Default fallback
    
    if rope_key in state_dict:
        # shape is [max_seq_len, dim] where dim = head_dim / 2
        rope_cos = state_dict[rope_key]
        max_seq_len = rope_cos.shape[0]
        rope_dim = rope_cos.shape[1]
        head_dim = rope_dim * 2
        num_heads = d_model // head_dim
    else:
        # Fallback: assume head_dim=64 if typical
        head_dim = 64
        num_heads = d_model // head_dim
        print(f"Warning: Could not find RoPE weights to infer head_dim/seq_len. Assuming head_dim={head_dim}, seq_len={max_seq_len}.")

    print(f"Inferred Config: vocab_size={vocab_size}, d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}, max_seq_len={max_seq_len}")
    return {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'max_seq_len': max_seq_len,
    }

def generate(
    checkpoint_path: str,
    tokenizer_path: str, # Directory containing vocab.json and merges.txt
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle if checkpoint is wrapped in a dict
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    config = infer_model_config(state_dict)

    print("Initializing model...")
    model = TransformerLM(**config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loading tokenizer from {tokenizer_path}...")
    vocab_path = os.path.join(tokenizer_path, 'vocab.json')
    merges_path = os.path.join(tokenizer_path, 'merges.txt')
    tokenizer = Tokenizer.from_files(vocab_path, merges_path)

    print("Generating...")
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # [1, seq_len]

    for _ in range(max_new_tokens):
        # Crop context if it exceeds max_seq_len (not strictly necessary with RoPE dynamic resize but good practice)
        # Here we just pass the full context.
        
        with torch.no_grad():
            logits = model(tokens)
        
        # Take the logits for the last token
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        tokens = torch.cat((tokens, next_token), dim=1)
        
        # Print the new token immediately (streaming effect)
        # Note: Tokenizer.decode might act weird with partial bytes, but usually fine for whole tokens
        new_word = tokenizer.decode([next_token.item()])
        print(new_word, end='', flush=True)

    print("\n\n--- Generation Complete ---")
    full_text = tokenizer.decode(tokens[0].tolist())
    # print(f"Full text:\n{full_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using Forge Transformer")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument('--prompt', type=str, default="Once upon a time", help="Prompt to start generation")
    parser.add_argument('--tokenizer_dir', type=str, default="bpe_model", help="Directory containing vocab.json and merges.txt")
    parser.add_argument('--max_tokens', type=int, default=200, help="Maximum new tokens to generate")
    parser.add_argument('--temp', type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    generate(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp
    )
