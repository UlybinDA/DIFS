import numpy as np
from services.circ_link_obst_i import vecs_pass_trough_circle_i
from services.circ_link_obst_n import vecs_pass_through_circle
from services.rctngl_link_obst import vecs_pass_through_rectangle

def check_circle_intersection(
        diff_vectors: np.ndarray,
        circle_normals: np.ndarray,
        origin_to_center: np.ndarray,
        diameter: float
) -> np.ndarray:
    return vecs_pass_trough_circle_i(
        np.asarray(diff_vectors, dtype=np.float64),
        np.asarray(circle_normals, dtype=np.float64),
        np.asarray(origin_to_center, dtype=np.float64),
        float(diameter)
    ).astype(bool)


def check_circle_angle(
        diff_vectors: np.ndarray,
        circle_normals: np.ndarray,
        max_ang_cos: float
) -> np.ndarray:
    return vecs_pass_through_circle(
        np.asarray(diff_vectors, dtype=np.float64),
        np.asarray(circle_normals, dtype=np.float64),
        float(max_ang_cos)
    ).astype(bool)


def check_rectangle_intersection(
        diff_vectors: np.ndarray,
        rectangle_vertices: np.ndarray
) -> np.ndarray:
    return vecs_pass_through_rectangle(
        np.asarray(diff_vectors, dtype=np.float64),
        np.asarray(rectangle_vertices, dtype=np.float64),
    ).astype(bool)