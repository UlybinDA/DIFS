# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy as np
from cython cimport boundscheck, wraparound
cimport cython
from libc.math cimport cos, sin

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_rotation_matrix(
    np.ndarray[np.float64_t, ndim=1] vector,
    np.ndarray[np.float64_t, ndim=1] angles,
    np.ndarray[np.float64_t, ndim=2] matr1,
    np.ndarray[np.float64_t, ndim=2] matr3,
    int axis,
    double direction,
):
    cdef Py_ssize_t n = angles.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty((n, 3), dtype=np.float64)

    cdef int i, j, k
    cdef double a
    cdef double R[3][3]
    cdef double tmp1[3]
    cdef double tmp2[3]

    for i in range(n):
        a = angles[i] * direction

        if axis == 0:  # x
            R[0][0], R[0][1], R[0][2] = 1, 0, 0
            R[1][0], R[1][1], R[1][2] = 0, cos(a), -sin(a)
            R[2][0], R[2][1], R[2][2] = 0, sin(a), cos(a)
        elif axis == 1:  # y
            R[0][0], R[0][1], R[0][2] = cos(a), 0, sin(a)
            R[1][0], R[1][1], R[1][2] = 0, 1, 0
            R[2][0], R[2][1], R[2][2] = -sin(a), 0, cos(a)
        elif axis == 2:  # z
            R[0][0], R[0][1], R[0][2] = cos(a), -sin(a), 0
            R[1][0], R[1][1], R[1][2] = sin(a), cos(a), 0
            R[2][0], R[2][1], R[2][2] = 0, 0, 1
        else:
            raise ValueError("Invalid axis")


        for j in range(3):
            tmp1[j] = 0
            for k in range(3):
                tmp1[j] += matr3[j, k] * vector[k]

        for j in range(3):
            tmp2[j] = 0
            for k in range(3):
                tmp2[j] += R[j][k] * tmp1[k]

        for j in range(3):
            result[i, j] = 0
            for k in range(3):
                result[i, j] += matr1[j, k] * tmp2[k]


    return result
