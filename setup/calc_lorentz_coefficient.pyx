# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound
cimport cython
from libc.math cimport sqrt

def calc_lorentz_coefficient(
    np.ndarray[np.float64_t, ndim=1] rotation_axis,
    np.ndarray[np.float64_t, ndim=2] diff_vectors,
    np.ndarray[np.float64_t, ndim=2] rec_vectors,
    np.ndarray[np.float64_t, ndim=1] beam_vec
    ):

    cdef Py_ssize_t n = diff_vectors.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] lor_coef = np.empty(n, dtype=np.float64)
    cdef double sin_theta, rec_vec_len, cos_gm, ksi, cos_mu, cos_nu
    cdef Py_ssize_t i
    cdef double rx, ry, rz
    cdef double dot1, dot2, dot3

    with nogil, parallel():
        for i in prange(n):
            rx = rec_vectors[i, 0]
            ry = rec_vectors[i, 1]
            rz = rec_vectors[i, 2]
            rec_vec_len = sqrt(rx*rx + ry*ry + rz*rz)
            rx = rx/rec_vec_len
            ry = ry/rec_vec_len
            rz = rz/rec_vec_len

            dot1 = beam_vec[0]*diff_vectors[i, 0] + beam_vec[1]*diff_vectors[i, 1] + beam_vec[2]*diff_vectors[i, 2]
            sin_theta = sqrt((1 - dot1) / 2.0)

            dot2 = rotation_axis[0]*rx + rotation_axis[1]*ry + rotation_axis[2]*rz
            cos_gm = sqrt(1 - dot2*dot2)

            dot3 = rotation_axis[0]*beam_vec[0] + rotation_axis[1]*beam_vec[1] + rotation_axis[2]*beam_vec[2]
            cos_mu = sqrt(1 - dot3*dot3)

            dot3 = rotation_axis[0]*diff_vectors[i, 0] + rotation_axis[1]*diff_vectors[i, 1] + rotation_axis[2]*diff_vectors[i, 2]
            cos_nu = sqrt(1 - dot3*dot3)

            ksi = 2.0 * sin_theta * cos_gm
            lor_coef[i] = calc_lorentz(ksi, cos_mu, cos_nu)

    return lor_coef


cdef inline double calc_lorentz(double ksi, double c_mu, double c_nu) nogil:
    return 0.5 * sqrt((ksi + c_mu + c_nu) *
                      (-ksi + c_mu + c_nu) *
                      (ksi - c_mu + c_nu) *
                      (ksi + c_mu - c_nu))