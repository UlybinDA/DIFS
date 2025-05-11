from rotation import apply_rotation_matrix
import numpy as np




def apply_rotation(rotations, no_of_scan, hkl_rotated, angles, matr1, matr3, directions, wavelength):
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
    axis = {'x': 0, 'y': 1, 'z': 2}[axis_char]
    return apply_rotation_matrix(
        vectors=hkl_rotated,
        angles=angles[:, 0],
        matr1=matr1,
        matr3=matr3,
        axis=axis,
        direction=directions[no_of_scan],
        inv_wavelength=1.0 / wavelength
    )
