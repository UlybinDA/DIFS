
#max index 500
def encode_hkl(array):
    base = 1001
    array = array.astype(int)
    encoded_array = array[:, 0] + array[:, 1] * base + array[:, 2] * base ** 2
    return encoded_array