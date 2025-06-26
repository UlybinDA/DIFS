from obstacles.obstacle import Ray_obstacle
import numpy as np
from sample.sample import Sample
from typing import Tuple, Union, Optional, List, Dict, Any
from nptyping import NDArray, Shape, Float, Int
from services.angle_calc import ang_bw_two_vects
from services.rotation_wrapper import apply_rotation_vec
from services.exceptions.exceptions import DiamondAnvilError
from assets.modals_content import *

class DiamondAnvil(Ray_obstacle):
    def __init__(self,
                 normal: Union[NDArray[Shape["1, 3"], Float], NDArray[Shape["1, 3"], Int]],
                 aperture: float):
        if aperture >= 90 or aperture <= 0:
            raise DiamondAnvilError(modal=aperture_value_error)

        self.normal = normal
        self.aperture = aperture

    def filter_anvil(self,
                     diff_vecs: np.ndarray,
                     diff_angles: np.ndarray,
                     rotation_axes: str,
                     directions_axes: Tuple[int, ...],
                     initial_axes_angles: Tuple[float, ...],
                     scan_axis_index: int,
                     data: Tuple[np.ndarray, ...],
                     mode: str,
                     incident_beam: np.ndarray[float]):
        cos_max_ang = np.cos(np.deg2rad(self.aperture))
        mode = 'false' if mode == 'shade' else 'true' if mode == 'transmit' else 'both' if mode == 'separate' else None
        incident_beam = incident_beam / np.linalg.norm(incident_beam).reshape(-1)
        matr1, matr3 = Sample.generate_rotation_matrices(rotations=rotation_axes, directions=directions_axes,
                                                         angle=initial_axes_angles, no_of_scan=scan_axis_index)
        rotation_axis = rotation_axes[scan_axis_index]
        direction = directions_axes[scan_axis_index]
        anvil_normals = apply_rotation_vec(vector=self.normal, angles=diff_angles, matr1=matr1, matr3=matr3,
                                           axis=rotation_axis, direction=direction)
        cos_bw_incident_norm = ang_bw_two_vects(vec1=anvil_normals, vec2=incident_beam, type='array', result='cos')
        cos_bw_diff_norm = (anvil_normals[:, 0] * diff_vecs[:, 0] + anvil_normals[:, 1] * diff_vecs[:, 1] +
                            anvil_normals[:, 2] * diff_vecs[:, 2])
        mask1 = (np.abs(cos_bw_diff_norm) >= cos_max_ang)
        mask2 = (np.abs(cos_bw_incident_norm) >= cos_max_ang)
        mask = (mask1 & mask2).reshape(-1, 1)

        data_output = tuple()
        for array in data:
            data_output += self.array_slice(array, mask, mode)
        return data_output

    def check_all_possible_anvil(self,
                                 ub_matr: np.ndarray,
                                 hkl: np.ndarray,
                                 data: Tuple[np.ndarray, ...],
                                 wavelength: float,
                                 separate_back=False) -> Dict[str, List[np.ndarray]]:

        hkl = hkl.reshape(-1, 3, 1)
        aperture_rad = np.deg2rad(self.aperture)
        s = np.matmul(ub_matr, hkl).reshape(-1, 3)
        alpha_ang = np.arctan(-s[:, 2] / s[:, 1])
        alpha_ang[np.isnan(alpha_ang)] = 0
        s_ = s
        s_[:, 0] = np.abs(s[:, 0])
        s_[:, 1] = np.abs(np.cos(alpha_ang) * s[:, 1] - np.sin(alpha_ang) * s[:, 2])
        s_[:, 2] = 0
        mask2lambda = (np.linalg.norm(s_, axis=1).reshape(-1, 1) <= 2 / wavelength).reshape(-1, 1)
        mask_front = ((wavelength * s_[:, 0] + np.cos(aperture_rad)) ** 2 + (
                wavelength * s_[:, 1] - np.sin(aperture_rad)) ** 2 <= 1).reshape(-1, 1) & mask2lambda
        mask_rear = (s_[:, 0] > 2 * np.cos(aperture_rad) / wavelength).reshape(-1, 1) & mask2lambda

        if separate_back:
            data_front = []
            for array in data:
                data_front += self.array_slice(array, mask_front, 'true')
            data_rear = []
            for array in data:
                data_rear += self.array_slice(array, mask_rear, 'true')
            return {'double_window': data_front, 'single_window': data_rear}
        else:
            data_all = []
            mask_all = mask_front | mask_rear
            for array in data:
                data_all += self.array_slice(array, mask_all, 'true')
            return {'all_windows': data_all}


