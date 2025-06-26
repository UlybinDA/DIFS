import numpy as np
from typing import Union



def ang_bw_two_vects(vec1: np.ndarray,
                     vec2: np.ndarray,
                     type: str = 'list',
                     result: str = 'angle') -> Union[float, np.ndarray]:
    if type == 'list':
        dot_prod = np.dot(vec1, vec2)
        if result == 'angle':
            angle = np.arccos(dot_prod / np.linalg.norm(vec1) / np.linalg.norm(vec2))
            return angle
        elif result == 'cos':
            cos = dot_prod / np.linalg.norm(vec1) / np.linalg.norm(vec2)
            return cos

    if type == 'array':
        if result == 'angle':
            angle_array = np.arccos((vec1[:, 0] * vec2[0] + vec1[:, 1] * vec2[1] + vec1[:, 2] * vec2[2]) / (
                    vec1[:, 0] ** 2 + vec1[:, 1] ** 2 + vec1[:, 2] ** 2) ** 0.5 / np.linalg.norm(vec2))
            return angle_array
        elif result == 'cos':
            cos_array = (vec1[:, 0] * vec2[0] + vec1[:, 1] * vec2[1] + vec1[:, 2] * vec2[2]) / (
                    vec1[:, 0] ** 2 + vec1[:, 1] ** 2 + vec1[:, 2] ** 2) ** 0.5 / np.linalg.norm(vec2)
            return cos_array
    return None