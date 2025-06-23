# cython: c_string_type=bytes, c_string_encoding=ascii
# cython: infer_types=True
# cython: overflowcheck=False
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

cdef double EPS = 1e-12

@cython.boundscheck(False)
@cython.wraparound(False)
def vecs_pass_trough_circle_i(
    np.ndarray[np.float64_t, ndim=2] diff_vectors,
    np.ndarray[np.float64_t, ndim=2] circle_normals,
    np.ndarray[np.float64_t, ndim=2] vec_origin_to_centre,
    double diameter,
):
    cdef double[:, ::1] diff_view = np.ascontiguousarray(diff_vectors, dtype=np.float64)
    cdef double[:, ::1] circle_norm_view = np.ascontiguousarray(circle_normals, dtype=np.float64)
    cdef double[:, ::1] origin_centre_view = np.ascontiguousarray(vec_origin_to_centre, dtype=np.float64)

    cdef Py_ssize_t n = diff_view.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] check = np.zeros(n, dtype=np.uint8)
    cdef unsigned char[:] check_view = check

    cdef double n_dot_v, n_dot_c, t, r_sq
    cdef double px, py, pz, dx, dy, dz, dist_sq
    cdef double cx, cy, cz, vx, vy, vz, nx, ny, nz
    cdef double cross_x, cross_y, cross_z, v_norm_sq, cross_norm_sq
    cdef Py_ssize_t i

    r_sq = (diameter / 2.0) * (diameter / 2.0)

    for i in prange(n, nogil=True):
        cx = origin_centre_view[i, 0]
        cy = origin_centre_view[i, 1]
        cz = origin_centre_view[i, 2]

        vx = diff_view[i, 0]
        vy = diff_view[i, 1]
        vz = diff_view[i, 2]

        nx = circle_norm_view[i, 0]
        ny = circle_norm_view[i, 1]
        nz = circle_norm_view[i, 2]

        n_dot_v = nx * vx + ny * vy + nz * vz
        n_dot_c = nx * cx + ny * cy + nz * cz

        if fabs(n_dot_v) < EPS:
            if fabs(n_dot_c) < EPS:
                cross_x = cy * vz - cz * vy
                cross_y = cz * vx - cx * vz
                cross_z = cx * vy - cy * vx
                cross_norm_sq = cross_x*cross_x + cross_y*cross_y + cross_z*cross_z
                v_norm_sq = vx*vx + vy*vy + vz*vz

                if v_norm_sq < EPS:
                    dist_sq = cx*cx + cy*cy + cz*cz
                    check_view[i] = 1 if dist_sq <= r_sq else 0
                else:
                    check_view[i] = 1 if cross_norm_sq <= r_sq * v_norm_sq else 0
            else:
                check_view[i] = 0

        else:
            t = n_dot_c / n_dot_v
            if t < 0:
            else:
                px = t * vx
                py = t * vy
                pz = t * vz

                dx = px - cx
                dy = py - cy
                dz = pz - cz

                dist_sq = dx*dx + dy*dy + dz*dz
                check_view[i] = 1 if dist_sq <= r_sq else 0

    return np.asarray(check)

cdef double fabs(double x) nogil:
    return -x if x < 0 else x