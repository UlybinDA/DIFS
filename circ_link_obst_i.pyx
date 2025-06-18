# cython: c_string_type=bytes, c_string_encoding=ascii
# cython: infer_types=True
# cython: overflowcheck=False
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def vecs_pass_trough_circle_i(
    np.ndarray[np.float64_t, ndim=2] diff_vectors,
    np.ndarray[np.float64_t, ndim=2] circle_vectors,
    np.ndarray[np.float64_t, ndim=2] circle_normals,
    np.ndarray[np.float64_t, ndim=1] vec_origin_to_centre,
    double diameter,
):
    cdef double[:, ::1] diff_view = np.ascontiguousarray(diff_vectors, dtype=np.float64)
    cdef double[:, ::1] circle_vec_view = np.ascontiguousarray(circle_vectors, dtype=np.float64)
    cdef double[:, ::1] circle_norm_view = np.ascontiguousarray(circle_normals, dtype=np.float64)
    cdef double[::1] origin_centre_view = np.ascontiguousarray(vec_origin_to_centre, dtype=np.float64)
    
    cdef Py_ssize_t n = diff_view.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] check = np.zeros(n, dtype=np.uint8)
    cdef unsigned char[:] check_view = check
    
    cdef double scalar_sum, D, denom
    cdef double x_intersection, y_intersection, z_intersection
    cdef double x_intersection_, y_intersection_, z_intersection_
    cdef unsigned char check0
    
    with nogil:
        for i in range(n):
            scalar_sum = (
                diff_view[i, 0] * circle_norm_view[i, 0] +
                diff_view[i, 1] * circle_norm_view[i, 1] +
                diff_view[i, 2] * circle_norm_view[i, 2]
            )
            
            D = -(
                circle_vec_view[i, 0] * circle_norm_view[i, 0] +
                circle_vec_view[i, 1] * circle_norm_view[i, 1] +
                circle_vec_view[i, 2] * circle_norm_view[i, 2]
            )
            
            denom = scalar_sum
            

            x_intersection = -D * diff_view[i, 0] / denom
            y_intersection = -D * diff_view[i, 1] / denom
            z_intersection = -D * diff_view[i, 2] / denom
            
            x_intersection_ = x_intersection - origin_centre_view[0]
            y_intersection_ = y_intersection - origin_centre_view[1]
            z_intersection_ = z_intersection - origin_centre_view[2]
            
            check0 = scalar_sum < 0.0
            check_view[i] = (
                (x_intersection_ * x_intersection_ +
                 y_intersection_ * y_intersection_ +
                 z_intersection_ * z_intersection_) < (diameter * diameter / 4.0)
            ) and check0
    
    return np.asarray(check)