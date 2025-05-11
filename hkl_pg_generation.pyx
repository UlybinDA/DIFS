import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound
from cython.parallel import prange


cdef int BASE = 1001  # hkl [-500, 500] range is enough
cdef long BASE_SQ = BASE * BASE

@boundscheck(False)
@wraparound(False)
cdef inline np.int32_t encode_single(int h, int k, int l) nogil:
    return h + k * BASE + l * BASE_SQ

@boundscheck(False)
@wraparound(False)
def encode_hkl_cython(np.int32_t[:, :] array_view):
    cdef int n = array_view.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] encoded = np.empty(n, dtype=np.int32)
    cdef np.int32_t[:] encoded_view = encoded

    cdef int i
    for i in prange(n, nogil=True, schedule='guided'):
        encoded_view[i] = encode_single(
            array_view[i, 0],
            array_view[i, 1],
            array_view[i, 2]
        )
    return encoded

@boundscheck(False)
@wraparound(False)
def generate_hkl_symmetry_cython(
    np.float32_t[:, :] hkl_orig_view,
    np.float64_t[:, :, :] positions_view
):
    cdef int N = hkl_orig_view.shape[0]
    cdef int num_pos = positions_view.shape[0]
    cdef int total = N * (num_pos + 1)

    cdef np.ndarray[np.float32_t, ndim=2] hkl_array = np.zeros((total, 3), dtype=np.float32)
    cdef np.float32_t[:, :] hkl_view = hkl_array

    cdef int i
    for i in prange(N, nogil=True):
        hkl_view[i, 0] = hkl_orig_view[i, 0]
        hkl_view[i, 1] = hkl_orig_view[i, 1]
        hkl_view[i, 2] = hkl_orig_view[i, 2]

    cdef int j, offset
    cdef np.float32_t pos00, pos01, pos02, pos10, pos11, pos12, pos20, pos21, pos22
    for j in prange(num_pos, nogil=True, schedule='dynamic'):
        offset = (j + 1) * N
        pos00 = positions_view[j, 0, 0]
        pos01 = positions_view[j, 0, 1]
        pos02 = positions_view[j, 0, 2]
        pos10 = positions_view[j, 1, 0]
        pos11 = positions_view[j, 1, 1]
        pos12 = positions_view[j, 1, 2]
        pos20 = positions_view[j, 2, 0]
        pos21 = positions_view[j, 2, 1]
        pos22 = positions_view[j, 2, 2]



        for i in range(N):
            hkl_view[offset + i, 0] = pos00 * hkl_orig_view[i, 0] + pos01 * hkl_orig_view[i, 1] + pos02 * hkl_orig_view[i, 2]
            hkl_view[offset + i, 1] = pos10 * hkl_orig_view[i, 0] + pos11 * hkl_orig_view[i, 1] + pos12 * hkl_orig_view[i, 2]
            hkl_view[offset + i, 2] = pos20 * hkl_orig_view[i, 0] + pos21 * hkl_orig_view[i, 1] + pos22 * hkl_orig_view[i, 2]
    cdef np.ndarray[np.float32_t, ndim=2] hkl_rounded = np.round(hkl_view).astype(np.float32)
    cdef np.ndarray[np.int32_t, ndim=2] hkl_int_array = hkl_rounded.astype(np.int32)
    cdef np.int32_t[:, :] hkl_int_view = hkl_int_array
    encoded = encode_hkl_cython(hkl_int_view)
    _, unique_indices = np.unique(encoded, return_index=True)

    return (
        hkl_array[unique_indices].astype(np.int32),
        np.tile(hkl_orig_view, (num_pos + 1, 1))[unique_indices],
    )

