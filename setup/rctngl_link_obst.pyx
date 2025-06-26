# cython: c_string_type=bytes, c_string_encoding=ascii
# cython: infer_types=True
# cython: overflowcheck=False
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from libc.math cimport sqrt
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def vecs_pass_through_rectangle(
    np.ndarray[np.float64_t, ndim=2] diff_vectors,
    np.ndarray[np.float64_t, ndim=3] rectangle_vectors_in,
):
    cdef double[:, ::1] diff_view = np.ascontiguousarray(diff_vectors, dtype=np.float64)
    cdef double[:, :, ::1] rect_view = np.ascontiguousarray(rectangle_vectors_in, dtype=np.float64)

    cdef Py_ssize_t n = diff_view.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] check = np.zeros(n, dtype=np.uint8)
    cdef unsigned char[:] check_view = check

    cdef double normal[3]

    cdef unsigned char check0, check1, check2, check3

    cdef Py_ssize_t i
    with nogil:
        for i in range(n):
            cross_prod_nogil(&rect_view[0, i, 0], &rect_view[1, i, 0], normal)
            check0 = (diff_view[i, 0] * normal[0] + diff_view[i, 1] * normal[1] + diff_view[i, 2] * normal[2]) > 0

            cross_prod_nogil(&rect_view[1, i, 0], &rect_view[2, i, 0], normal)
            check1 = (diff_view[i, 0] * normal[0] + diff_view[i, 1] * normal[1] + diff_view[i, 2] * normal[2]) > 0

            cross_prod_nogil(&rect_view[2, i, 0], &rect_view[3, i, 0], normal)
            check2 = (diff_view[i, 0] * normal[0] + diff_view[i, 1] * normal[1] + diff_view[i, 2] * normal[2]) > 0

            cross_prod_nogil(&rect_view[3, i, 0], &rect_view[0, i, 0], normal)
            check3 = (diff_view[i, 0] * normal[0] + diff_view[i, 1] * normal[1] + diff_view[i, 2] * normal[2]) > 0

            check_view[i] = check0 & check1 & check2 & check3

    return np.asarray(check_view)

cdef inline void cross_prod_nogil(
    const double* vec1,
    const double* vec2,
    double* out
) nogil:
    out[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    out[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    out[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]