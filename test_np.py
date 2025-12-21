import numpy as np
a = np.load("data/TinyStories/train_tokens_uint16.npy", mmap_mode="r")
print(a.dtype, a.shape)
print(a[:20])

