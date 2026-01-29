import numpy as np

#max index 500
def encode_hkl(array):
    base = 1001
    arr = np.rint(array).astype(np.int32, copy=False)
    encoded_array = arr[:, 0] + arr[:, 1] * base + arr[:, 2] * base * base
    return encoded_array.reshape(-1,1)