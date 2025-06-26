from obstacles.obstacle import Ray_obstacle
import numpy as np
from sample.sample import Sample
from typing import Tuple, Union, Optional, List, Dict, Any
from nptyping import NDArray, Shape, Float, Int
from services.angle_calc import ang_bw_two_vects
from services.rotation_wrapper import apply_rotation_vec
from services.obst_check_wrapper import check_rectangle_intersection, check_circle_intersection, check_circle_angle
from scipy.spatial.transform import Rotation as R


class LinkedObstacle(Ray_obstacle):
    def __init__(self,
                 highest_linked_axis_index: int,
                 dist: float,
                 geometry: str,
                 disp_y: float,
                 disp_z: float,
                 rot: np.ndarray,
                 orientation: str,
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 diameter: Optional[float] = None,
                 name: str = ''
                 ):
        super().__init__(dist=dist, geometry=geometry, disp_y=disp_y, disp_z=disp_z, rot=rot, orientation=orientation,
                         height=height, width=width, diameter=diameter, complex=False)
        self.highest_linked_axis_index = highest_linked_axis_index

    def _zero_angle_rotation(self):
        init_rotation = R.from_euler('xyz', angles=self.rot, degrees=True)
        vecs = self._prepare_obstacle_vecs(orientation=self.orientation, geometry=self.geometry,
                                           height=self.height, rotation=init_rotation, width=self.width,
                                           diameter=self.diameter,
                                           vec_origin_to_centre=self.vec_origin_to_centre, disp_y=self.disp_y,
                                           disp_z=self.disp_z)
        return vecs

    def _static_angle_filter(self, vecs, axes_linked_to_obst, data, diff_vecs, mode, angles, directions):
        angles *= directions
        linked_rotation = R.from_euler(axes_linked_to_obst, angles=angles, degrees=True)
        vecs = linked_rotation.apply(vecs)
        data = self.filter(diff_vecs=diff_vecs, data=data, mode=mode, vecs=vecs)
        return data

    def filter_linked_obstacle(self, scan_axis_index, diff_vectors, initial_axes_angles, diff_angles, directions_axes,
                               data, rotation_axes,
                               mode):
        mode = 'false' if mode == 'shade' else 'true' if mode == 'transmit' else 'both' if mode == 'separate' else None
        static_filter = True if self.highest_linked_axis_index > scan_axis_index else False
        axes_linked_to_obst = rotation_axes[self.highest_linked_axis_index:]
        angles = initial_axes_angles[self.highest_linked_axis_index:]
        directions_axes = directions_axes[self.highest_linked_axis_index:]
        vecs = self._zero_angle_rotation()
        if static_filter:
            return self._static_angle_filter(vecs, axes_linked_to_obst, data, diff_vectors, mode, angles,
                                             directions_axes)
        else:
            return self._dynamic_angle_filter(scan_axis_index, axes_linked_to_obst, directions_axes, angles,
                                              rotation_axes, vecs, diff_angles,
                                              diff_vectors, data, mode)

    def _dynamic_angle_filter(self, scan_axis_index, axes_linked_to_obst, directions, angles, axes, vecs, diff_angles,
                              diff_vectors, data, mode):
        slice_shift_scan_axis = scan_axis_index - self.highest_linked_axis_index
        matr1, matr3 = Sample.generate_rotation_matrices(axes_linked_to_obst, directions, angles,
                                                         slice_shift_scan_axis)
        rotation_axis = axes[slice_shift_scan_axis]
        direction = directions[slice_shift_scan_axis]
        vecs_rotated = []

        if self.geometry == 'rectangle':
            if ang_bw_two_vects(np.cross(vecs[0], vecs[1]), vecs[2]) > np.pi / 2:
                vecs = vecs[::-1]

        if self.geometry == 'circle' and self.orientation == 'independent':
            vecs = np.vstack((vecs[1:, :], self.vec_origin_to_centre))

        for vec in vecs:
            vecs_rotated.append(apply_rotation_vec(vector=vec, angles=diff_angles, matr1=matr1, matr3=matr3,
                                                   axis=rotation_axis, direction=direction))

        vecs = np.array(vecs_rotated)
        if self.geometry == 'rectangle':
            check = check_rectangle_intersection(diff_vectors=diff_vectors, rectangle_vertices=vecs).reshape(-1, 1)
        else:
            if self.orientation == 'normal':
                max_ang_cos = ang_bw_two_vects(vec1=vecs[0, 0], vec2=vecs[1, 0], result='cos')
                check = check_circle_angle(diff_vectors=diff_vectors, circle_normals=vecs[0],
                                           max_ang_cos=max_ang_cos).reshape(-1, 1)
            else:
                check = check_circle_intersection(diff_vectors=diff_vectors,
                                                  circle_normals=vecs[0],
                                                  origin_to_center=vecs[1],
                                                  diameter=self.diameter).reshape(-1, 1)
        data_output = self._slice_data(data, check, mode)
        return data_output

    def create_obst_vec_arr(self,
                            obst_vecs: NDArray[Shape["1, 3"], Float],
                            diff_angles: NDArray,
                            rotation_axes: str,
                            directions_axes: Tuple[int, ...],
                            initial_axes_angles: Tuple[float, ...],
                            scan_axis_index: int
                            ):
        assert set(rotation_axes).issubset(['x', 'y', 'z']) and rotation_axes != '', 'Wrong rotation axes!'
        n_obst_vecs = obst_vecs.shape[0]
        n_angles = len(diff_angles)
        obst_vecs_ = np.tile(obst_vecs, (n_angles, 1, 1))
        rot_a = rotation_axes[scan_axis_index]
        rot_vecs = {'x': np.array([1, 0, 0]),
                    'y': np.array([0, 1, 0]),
                    'z': np.array([0, 0, 1])}
        rot_vec = rot_vecs.get(rot_a)
        rot_vecs = rot_vec * directions_axes
        rot_vecs = np.repeat(rot_vecs, repeats=n_obst_vecs, axis=0)
        rotations = R.from_rotvec(rot_vecs)
        obst_vecs_ = rotations.apply(obst_vecs_.reshape(-1, 3)).reshape(n_obst_vecs, -1, 3)
        return obst_vecs_
