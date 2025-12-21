import torch
import numpy as np


def get_batch(data, batch_size, context_len, device="cuda"):
    n = len(data)
    starts = np.random.randint(0, n - context_len - 1, size=(batch_size,))
    # 由于预测下一个 token  取 n - context_len - 1 位置作为样本起点

    x_batch = np.stack([data[i : i + context_len] for i in starts])
    y_batch = np.stack([data[i + 1 : i + 1 + context_len] for i in starts])

    input_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
    output_batch = torch.tensor(y_batch, dtype=torch.long, device=device)
    return input_batch, output_batch
