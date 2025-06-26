from services.rotation import apply_rotation_matrix
from services.vec_mx_rot import apply_vec_rotation
import numpy as np

axes_dict = {'x': 0, 'y': 1, 'z': 2}


def apply_rotation_vecs(rotations, no_of_scan, hkl_rotated, angles, matr1, matr3, directions, wavelength):
    assert isinstance(hkl_rotated, np.ndarray)
    assert isinstance(angles, np.ndarray)
    assert hkl_rotated.dtype == np.float64, f"Expected float64, got {hkl_rotated.dtype}"
    assert angles.dtype == np.float64, f"Expected float64, got {angles.dtype}"
    assert hkl_rotated.shape[0] == angles.shape[0], "Mismatch in number of vectors and angles"
    assert hkl_rotated.shape[1] == 3, "Vectors must be 3D"
    assert angles.ndim == 2 and angles.shape[1] == 1, "Angles must be of shape (N, 1)"
    assert matr1.shape == (3, 3)
    assert matr3.shape == (3, 3)

    axis_char = rotations[no_of_scan]
    axis = axes_dict[axis_char]
    return apply_rotation_matrix(
        vectors=hkl_rotated,
        angles=angles[:, 0],
        matr1=matr1,
        matr3=matr3,
        axis=axis,
        direction=directions[no_of_scan],
        inv_wavelength=1.0 / wavelength
    )


def apply_rotation_vec(vector, angles, matr1, matr3, axis, direction):
    assert isinstance(vector, np.ndarray)
    vector = vector.reshape(-1).astype(np.float64)
    angles = angles.reshape(-1).astype(np.float64)
    direction = float(direction)
    assert isinstance(angles, np.ndarray)
    assert isinstance(matr1, np.ndarray)
    assert isinstance(matr3, np.ndarray)
    assert isinstance(axis, str)
    assert vector.dtype == np.float64, f"Vector.dtype expected float64, got {vector.dtype}"
    assert axis in {'x', 'y', 'z'} and len(axis) == 1
    assert isinstance(direction, float)
    assert matr1.shape == (3, 3)
    assert matr3.shape == (3, 3)
    axis = axes_dict[axis]
    return apply_vec_rotation(
        vector,
        angles,
        matr1,
        matr3,
        axis,
        direction)
