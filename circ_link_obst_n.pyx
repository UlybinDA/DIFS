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
def vecs_pass_through_circle(
    np.ndarray[np.float64_t, ndim=2] diff_vectors,
    np.ndarray[np.float64_t, ndim=2] circle_normals,
    double max_ang_cos

):
    cdef double[:, ::1] diff_view = np.ascontiguousarray(diff_vectors, dtype=np.float64)
    cdef double[:, ::1] circle_norm_view = np.ascontiguousarray(circle_normals, dtype=np.float64)

    cdef Py_ssize_t n = diff_view.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] check = np.zeros(n, dtype=np.uint8)
    cdef unsigned char[:] check_view = check

    cdef double cos_
    with nogil:
        for i in range(n):
            cos_ = cos_bw_angles(diff_view[i, 0], diff_view[i, 1], diff_view[i, 2],
                    circle_norm_view[i, 0], circle_norm_view[i, 1], circle_norm_view[i, 2])

            check_view[i] = cos_ > max_ang_cos
    return np.asarray(check_view)

cdef double cos_bw_angles(
    double ax, double ay, double az,
    double bx, double by, double bz
) nogil:
    cdef double dot_product = ax * bx + ay * by + az * bz
    cdef double norm_a = ax * ax + ay * ay + az * az
    cdef double norm_b = bx * bx + by * by + bz * bz

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    cdef double norm_product = sqrt(norm_a * norm_b)
    cdef double cos_theta = dot_product / norm_product

    if cos_theta > 1.0:
        return 1.0
    elif cos_theta < -1.0:
        return -1.0
    else:
        return cos_theta