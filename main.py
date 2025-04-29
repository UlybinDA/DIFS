import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from io import StringIO
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Modals_content import *
from Exceptions import DiamondAnvilError
from pointsymmetry import generate_hkl_by_pg, PG_KEYS, get_key, generate_orig_hkl_array
from my_logger import mylogger
from typing import Tuple, Union, Optional, List, Dict, Any
from nptyping import NDArray, Shape, Float, Int

warnings.filterwarnings("ignore", category=RuntimeWarning)


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


class Ray_obstacle():
    @mylogger('DEBUG', log_args=True)
    def __init__(self,
                 dist: float,
                 geometry: str,
                 disp_y: float,
                 disp_z: float,
                 rot: np.ndarray,
                 orientation: str,
                 height: Optional[float] = None,
                 width: Optional[float] = None,
                 diameter: Optional[float] = None,
                 complex: bool = False,
                 complex_format: Tuple[int, int] = (5, 2),
                 row_col_spacing: Tuple[float, float] = (0.344, 2.924),
                 chip_thickness: Optional[float] = None,
                 dead_areas: Optional[List[Tuple[float, float]]] = None):

        self.dist = dist
        self.disp_y, self.disp_z = (disp_y, disp_z)
        self.geometry = geometry
        self.orientation = orientation
        self.diameter = diameter
        self.width = width
        self.height = height
        self.rot = rot
        self.chip_thickness = chip_thickness
        self.dead_areas = dead_areas

        if orientation == 'normal':
            self.vec_origin_to_centre = np.array([dist, 0, 0])
        else:
            self.vec_origin_to_centre = np.array([dist, disp_y, disp_z])

        if complex is True:
            self.complex = complex
            self.row_spacing, self.col_spacing = row_col_spacing
            self.chip_height = (height - (complex_format[0] - 1) * row_col_spacing[0]) / complex_format[0]
            self.chip_width = (width - (complex_format[1] - 1) * row_col_spacing[1]) / complex_format[1]
            self.complex_format = complex_format

    def vecs_complex_obstacle(self,
                              width: Optional[float] = None,
                              height: Optional[float] = None,
                              thickness: float = 0) -> np.ndarray:
        rotation = R.from_euler('xyz', self.rot, degrees=True)
        vecs = np.array([])
        cw, ch, hs, ws = (self.chip_width, self.chip_height, self.row_spacing, self.col_spacing)
        t = thickness
        if width or height is None:
            width, height = (self.width, self.height)

        if self.orientation == 'independent':
            for c in range(self.complex_format[1]):
                for r in range(self.complex_format[0]):
                    chip_vecs = np.array(
                        [[t, width / 2 - ws * c - cw * c, height / 2 - hs * r - ch * r],
                         [t, width / 2 - ws * c - cw * c, height / 2 - hs * r - ch * (r + 1)],
                         [t, width / 2 - ws * c - cw * (c + 1), height / 2 - hs * r - ch * (r + 1)],
                         [t, width / 2 - ws * c - cw * (c + 1), height / 2 - hs * r - ch * r]])
                    vecs = np.append(vecs, chip_vecs)
            vecs = vecs.reshape(-1, 3)
            vecs = rotation.apply(vecs)
            vecs = vecs + self.vec_origin_to_centre
            vecs = vecs.reshape(-1, 4, 3)

        elif self.orientation == 'normal':
            for r in range(self.complex_format[0]):
                for c in range(self.complex_format[1]):
                    chip_vecs = np.array(
                        [[t, width / 2 - ws * c - cw * c, height / 2 - hs * r - ch * r],
                         [t, width / 2 - ws * c - cw * c, height / 2 - hs * r - ch * (r + 1)],
                         [t, width / 2 - ws * c - cw * (c + 1), height / 2 - hs * r - ch * (r + 1)],
                         [t, width / 2 - ws * c - cw * (c + 1), height / 2 - hs * r - ch * r]])
                    vecs = np.append(vecs, chip_vecs)
            vecs = vecs.reshape(-1, 3)
            vecs = vecs + self.vec_origin_to_centre
            vecs = rotation.apply(vecs)
            vecs = vecs.reshape(-1, 4, 3)
        return vecs

    def filter_complex_obstacle(self,
                                diff_vecs: np.ndarray,
                                data: Tuple[np.ndarray, ...],
                                mode: str = 'transmit',
                                width: Optional[float] = None,
                                height: Optional[float] = None) -> Tuple[Tuple[np.ndarray, ...], ...]:

        if width or height is None:
            width, height = (self.width, self.height)
        rotation = R.from_euler('xyz', self.rot, degrees=True)

        if self.orientation == 'normal':
            det_vecs = np.array(
                [[0, width / 2, height / 2], [0, width / 2, -height / 2], [0, -width / 2, -height / 2],
                 [0, -width / 2, height / 2]])
            det_vecs = det_vecs + self.vec_origin_to_centre
            det_vecs = rotation.apply(det_vecs)

        elif self.orientation == 'independent':
            det_vecs = np.array(
                [[0, width / 2, height / 2], [0, width / 2, -height / 2], [0, -width / 2, -height / 2],
                 [0, -width / 2, height / 2]])
            det_vecs = rotation.apply(det_vecs)
            det_vecs = det_vecs + self.vec_origin_to_centre

        vecs = self.vecs_complex_obstacle()
        data_output = tuple()
        data = self.filter(diff_vecs=diff_vecs, data=data, vecs=det_vecs, mode=mode)

        for chip in range(vecs.shape[0]):
            single_chip_data = ((self.filter(diff_vecs=data[0], data=data, vecs=vecs[chip], mode=mode),),)
            data_output += single_chip_data
        return data_output

    def get_parallax_vecs(self,
                          chip_thickness: float,
                          width: Optional[float] = None,
                          height: Optional[float] = None,
                          diameter: Optional[float] = None,
                          orientation: Optional[str] = None,
                          vec_origin_to_centre: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:

        rotation = R.from_euler('xyz', self.rot, degrees=True)
        if orientation is None:
            orientation = self.orientation
        if vec_origin_to_centre is None:
            vec_origin_to_centre = self.vec_origin_to_centre

        if orientation == 'independent':
            if self.geometry == 'rectangle':
                vecs = np.array(
                    [[chip_thickness, width / 2, height / 2],
                     [chip_thickness, width / 2, -height / 2],
                     [chip_thickness, -width / 2, -height / 2],
                     [chip_thickness, -width / 2, height / 2],
                     [0, width / 2, height / 2],
                     [0, width / 2, -height / 2],
                     [0, -width / 2, -height / 2],
                     [0, -width / 2, height / 2]])
                vecs = rotation.apply(vecs)
                normals = np.array([[0, width / 2, 0], [chip_thickness, 0, 0], [0, 0, height / 2]])
                normals = rotation.apply(normals)
                vecs = vecs + vec_origin_to_centre
                return (vecs, normals)
            elif self.geometry == 'circle':
                vecs = rotation.apply(np.array([chip_thickness, diameter / 2, 0]))
                normals = rotation.apply(np.array([1, 0, 0]))
                back_center = vec_origin_to_centre + rotation.apply(np.array([chip_thickness, 0, 0]))
                vecs = vecs + vec_origin_to_centre
                return (vecs, normals, back_center)

        elif orientation == 'normal':
            if self.geometry == 'rectangle':
                vecs = np.array(
                    [[chip_thickness, width / 2, height / 2], [chip_thickness, width / 2, -height / 2],
                     [chip_thickness, -width / 2, -height / 2], [chip_thickness, -width / 2, height / 2]])
                normals = np.array([[0, width / 2, 0], [chip_thickness, 0, 0], [0, 0, height / 2]])
                normals = rotation.apply(normals)
                vecs = vecs + vec_origin_to_centre
                vecs = rotation.apply(vecs)
                return (vecs, normals)
            elif self.geometry == 'circle':
                vecs = rotation.apply(np.array([chip_thickness, diameter / 2, 0]) + vec_origin_to_centre)
                normals = rotation.apply(vec_origin_to_centre)
                back_center = rotation.apply(np.array([chip_thickness, 0, 0]) + vec_origin_to_centre)
                return (vecs, normals, back_center)

    def calf_inter_cylinder(self,
                            data: np.ndarray,
                            axis: np.ndarray,
                            diameter: Optional[float] = None) -> Dict[np.ndarray, np.ndarray]:

        if diameter is None:
            diameter = self.diameter
        R = diameter / 2
        a, b, c = (axis[0], axis[1], axis[2])
        x0, y0, z0 = (self.vec_origin_to_centre[0], self.vec_origin_to_centre[1], self.vec_origin_to_centre[2])
        e, f, g = (data[:, 0], data[:, 1], data[:, 2])
        sigma = a ** 2 + b ** 2 + c ** 2
        pi = -x0 * a - y0 * b - z0 * c
        alpha = e * a + f * b + g * c
        beta = e * x0 + f * y0 + g * z0
        gamma = x0 ** 2 + y0 ** 2 + z0 ** 2
        omega = e ** 2 + f ** 2 + g ** 2
        ro = omega - alpha ** 2 / sigma
        thetha = -2 * pi * alpha / sigma - 2 * beta
        tau = - pi ** 2 / sigma + gamma - (R ** 2)
        t1 = ((-thetha + (thetha ** 2 - 4 * ro * tau) ** 0.5) / 2 / ro)
        t2 = ((-thetha - (thetha ** 2 - 4 * ro * tau) ** 0.5) / 2 / ro)
        p1 = np.hstack(((e * t1).reshape(-1, 1), (f * t1).reshape(-1, 1), (g * t1).reshape(-1, 1)))
        p2 = np.hstack(((e * t2).reshape(-1, 1), (f * t2).reshape(-1, 1), (g * t2).reshape(-1, 1)))
        return (p1, p2)

    @staticmethod
    def vecs_bw_vecs4(vecs: np.ndarray,
                      vecs4: np.ndarray) -> np.ndarray:

        if ang_bw_two_vects(np.cross(vecs4[0], vecs4[1]), vecs4[2]) < np.pi / 2:
            normal = np.cross(vecs4[0], vecs4[1])
            check0 = (vecs[:, 0] * normal[0] + vecs[:, 1] * normal[1] + vecs[:, 2] * normal[2]) > 0
            normal = np.cross(vecs4[2], vecs4[3])
            check1 = (vecs[:, 0] * normal[0] + vecs[:, 1] * normal[1] + vecs[:, 2] * normal[2]) > 0
            normal = np.cross(vecs4[1], vecs4[2])
            check2 = (vecs[:, 0] * normal[0] + vecs[:, 1] * normal[1] + vecs[:, 2] * normal[2]) > 0
            normal = np.cross(vecs4[3], vecs4[0])
            check3 = (vecs[:, 0] * normal[0] + vecs[:, 1] * normal[1] + vecs[:, 2] * normal[2]) > 0
            check = (check0 & check1 & check2 & check3).reshape(-1, 1)
            return check

        elif ang_bw_two_vects(np.cross(vecs4[0], vecs4[1]), vecs4[2]) > np.pi / 2:
            normal = np.cross(vecs4[1], vecs4[0])
            check0 = (vecs[:, 0] * normal[0] + vecs[:, 1] * normal[1] + vecs[:, 2] * normal[2]) > 0
            normal = np.cross(vecs4[3], vecs4[2])
            check1 = (vecs[:, 0] * normal[0] + vecs[:, 1] * normal[1] + vecs[:, 2] * normal[2]) > 0
            normal = np.cross(vecs4[2], vecs4[1])
            check2 = (vecs[:, 0] * normal[0] + vecs[:, 1] * normal[1] + vecs[:, 2] * normal[2]) > 0
            normal = np.cross(vecs4[0], vecs4[3])
            check3 = (vecs[:, 0] * normal[0] + vecs[:, 1] * normal[1] + vecs[:, 2] * normal[2]) > 0
            check = (check0 & check1 & check2 & check3).reshape(-1, 1)
            return check

    def calf_inter_plane(self,
                         data: np.ndarray,
                         normal: np.ndarray,
                         point: np.ndarray) -> np.ndarray:

        i = -normal[0] * point[0] - normal[1] * point[1] - normal[2] * point[2]

        t = -i / (normal[0] * data[:, 0] + normal[1] * data[:, 1] + normal[2] * data[:, 2])
        x = (t * data[:, 0]).reshape(-1, 1)
        y = (t * data[:, 1]).reshape(-1, 1)
        z = (t * data[:, 2]).reshape(-1, 1)

        return np.hstack((x, y, z))

    def parallax_beta_complex(self,
                              diff_vecs: np.ndarray,
                              data: Tuple[np.ndarray, ...],
                              wavelength: float) -> Tuple[Tuple[np.ndarray, ...], ...]:

        rotation = R.from_euler('xyz', self.rot, degrees=True)
        vecs_front = self.vecs_complex_obstacle()
        vecs_back = self.vecs_complex_obstacle(thickness=self.chip_thickness)
        vecs = np.hstack((vecs_front, vecs_back))
        normals = np.array([[0, self.chip_width / 2, 0], [self.chip_thickness, 0, 0], [0, 0, self.chip_height / 2]])
        normals = rotation.apply(normals)

        vecs_origin_to_centre = np.array([])
        n_of_chips = vecs_front.shape[0]
        for i in range(n_of_chips):
            vecs_origin_to_centre = np.append(vecs_origin_to_centre,
                                              (vecs_front[i, 0, :] + vecs_front[i, 1, :]
                                               + vecs_front[i, 2, :] + vecs_front[i, 3, :]) / 4)
        vecs_origin_to_centre = vecs_origin_to_centre.reshape(-1, 3)
        data = self.filter_complex_obstacle(diff_vecs=diff_vecs, data=data, mode='transmit')
        data_output = tuple()
        for i in range(n_of_chips):
            newdata = ((self.parallax_beta(diff_vecs=data[i][0], data=data[i], normals=normals, vecs=vecs[i],
                                           wavelength=wavelength, chip_thickness=self.chip_thickness,
                                           vec_origin_to_centre=vecs_origin_to_centre[i], width=self.chip_width,
                                           height=self.chip_height),),)
            data_output += newdata
        return data_output

    def parallax_beta(self,
                      diff_vecs: np.ndarray,
                      data: Tuple[np.ndarray, ...],
                      chip_thickness: float,
                      wavelength: float,
                      vecs: Optional[np.ndarray] = None,
                      normals: Optional[np.ndarray] = None,
                      width: Optional[float] = None,
                      height: Optional[float] = None,
                      diameter: Optional[float] = None,
                      vec_origin_to_centre: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:

        with open('lin_att.txt', 'r') as file:
            lin_att = np.loadtxt(StringIO(file.read()))
            lin_att_coeff = lin_att[:, 1][np.argmin(np.abs(lin_att[:, 0] - wavelength))]

        if vecs is None or normals is None:
            if width or height is None and self.geometry == 'rectangle':
                width, height = (self.width, self.height)
            if diameter is None and self.geometry == 'circle':
                diameter = self.diameter
            if vec_origin_to_centre is None:
                vec_origin_to_centre = self.vec_origin_to_centre

            if self.geometry == 'rectangle':
                vecs, normals = self.get_parallax_vecs(chip_thickness, width, height,
                                                       vec_origin_to_centre=vec_origin_to_centre)
            elif self.geometry == 'circle':
                vecs, normals, back_center = self.get_parallax_vecs(chip_thickness, diameter=diameter,
                                                                    vec_origin_to_centre=vec_origin_to_centre)

        if self.geometry == 'rectangle':
            intersection = np.zeros(diff_vecs.shape[0]).reshape(-1, 1)

            check = self.vecs_bw_vecs4(vecs=diff_vecs, vecs4=np.array((vecs[0], vecs[1], vecs[5], vecs[4])))
            intersection[check == 1] = 1
            check = self.vecs_bw_vecs4(vecs=diff_vecs, vecs4=np.array((vecs[1], vecs[2], vecs[6], vecs[5])))
            intersection[check == 1] = 2
            check = self.vecs_bw_vecs4(vecs=diff_vecs, vecs4=np.array((vecs[2], vecs[6], vecs[7], vecs[3])))
            intersection[check == 1] = 3
            check = self.vecs_bw_vecs4(vecs=diff_vecs, vecs4=np.array((vecs[4], vecs[0], vecs[3], vecs[7])))
            intersection[check == 1] = 4

            sort_indices = intersection[:, 0].argsort()
            diff_vecs = np.hstack((diff_vecs, intersection))[sort_indices]
            data_output = tuple()
            for array in data:
                array = array[sort_indices]
                data_output += (array,)

            c = (
                np.count_nonzero(diff_vecs[:, 3] == 0), np.count_nonzero(diff_vecs[:, 3] == 1),
                np.count_nonzero(diff_vecs[:, 3] == 2),
                np.count_nonzero(diff_vecs[:, 3] == 3), np.count_nonzero(diff_vecs[:, 3] == 4))
            i = (c[0] + c[1], c[0] + c[1] + c[2], c[0] + c[1] + c[2] + c[3], c[0] + c[1] + c[2] + c[3] + c[4])
            xyzb = np.vstack((self.calf_inter_plane(diff_vecs[0:c[0], 0:3], normals[1], vecs[5]),
                              self.calf_inter_plane(diff_vecs[c[0]:i[0], 0:3], normals[0], vecs[0]),
                              self.calf_inter_plane(diff_vecs[i[0]:i[1], 0:3], normals[2], vecs[1]),
                              self.calf_inter_plane(diff_vecs[i[1]:i[2], 0:3], normals[0], vecs[2]),
                              self.calf_inter_plane(diff_vecs[i[2]:i[3], 0:3], normals[2], vecs[3])))
            xyzf = self.calf_inter_plane(diff_vecs[:, 0:3], normals[1], vec_origin_to_centre)
            path = xyzf - xyzb
            path_len = np.sqrt(path[:, 0] ** 2 + path[:, 1] ** 2 + path[:, 2] ** 2)
            intensity = np.exp(-lin_att_coeff * path_len).reshape(-1, 1)
            data_output += (intensity,)
            return data_output

        elif self.geometry == 'circle':
            data = self.filter(diff_vecs=diff_vecs, data=data, mode='separate', vecs=vecs, diameter=self.diameter,
                               intersection_coords=True, normal=normals, vec_origin_to_centre=back_center)
            diff_vecs = np.vstack((data[0], data[1]))
            face_intersection = self.filter(diff_vecs=diff_vecs, data=data, mode='separate', intersection_coords=True)[
                -1]
            wall_intersection = self.calf_inter_cylinder(data[1], normals)[0]
            bw_intersection = np.vstack((data[-1], wall_intersection))
            path = face_intersection - bw_intersection
            path_len = (path[:, 0] ** 2 + path[:, 1] ** 2 + path[:, 2] ** 2) ** 0.5
            intensity = np.exp(-lin_att_coeff * path_len).reshape(-1, 1)
            back_wall_marker = np.vstack((np.zeros((data[0].shape[0], 1)), np.ones((data[1].shape[0], 1))))
            data_output = data + (intensity, back_wall_marker)
            return data_output

    def array_slice(self,
                    array: np.ndarray,
                    boolarray: np.ndarray,
                    bool: str) -> Tuple[np.ndarray, ...]:

        shape = array.shape[1]
        if bool == 'true':
            data = (array[boolarray[:, 0] == True].reshape(-1, shape),)
        elif bool == 'false':
            data = (array[boolarray[:, 0] == False].reshape(-1, shape),)
        elif bool == 'both':
            data = (
                array[boolarray[:, 0] == True].reshape(-1, shape), array[boolarray[:, 0] == False].reshape(-1, shape))
        else:
            print('incorrect mode in slicing function')
        return data

    @mylogger('DEBUG')
    def filter(self,
               diff_vecs: np.ndarray,
               data: Tuple[np.ndarray, ...],
               mode: str,
               width: Optional[float] = None,
               height: Optional[float] = None,
               diameter: Optional[float] = None,
               vecs: Optional[np.ndarray] = None,
               intersection_coords: bool = False,
               normal: Optional[np.ndarray] = None,
               vec_origin_to_centre: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:

        if width == None and height == None and diameter == None and vec_origin_to_centre is None:
            width, height, diameter, vec_origin_to_centre = (
                self.width, self.height, self.diameter, self.vec_origin_to_centre)
        rotation = R.from_euler('xyz', self.rot, degrees=True)
        mode = 'false' if mode == 'shade' else 'true' if mode == 'transmit' else 'both' if mode == 'separate' else None

        if vecs is None:
            if self.orientation == 'independent':
                if self.geometry == 'rectangle':
                    vecs = np.array(
                        [[0, width / 2, height / 2], [0, width / 2, -height / 2], [0, -width / 2, -height / 2],
                         [0, -width / 2, height / 2]])
                    vecs = rotation.apply(vecs)
                    vecs = vecs + self.vec_origin_to_centre
                elif self.geometry == 'circle':
                    vecs = rotation.apply(np.array([0, diameter / 2, 0]))
                    normal = rotation.apply(np.array([1, 0, 0]))
                    vecs = vecs + self.vec_origin_to_centre

            elif self.orientation == 'normal':
                if self.disp_y or self.disp_z != 0:
                    print('disp y or z is not 0 if its made intentionally switch to independent orientation')
                if self.geometry == 'rectangle':
                    vecs = np.array(
                        [[0, width / 2, height / 2], [0, width / 2, -height / 2], [0, -width / 2, -height / 2],
                         [0, -width / 2, height / 2]])
                    vecs = vecs + self.vec_origin_to_centre
                    vecs = rotation.apply(vecs)
                elif self.geometry == 'circle':
                    vecs = np.array([[0, 0, 0], [0, 0, diameter / 2]]) + self.vec_origin_to_centre
                    vecs = rotation.apply(vecs)

        if self.geometry == 'rectangle':
            if ang_bw_two_vects(np.cross(vecs[0], vecs[1]), vecs[2]) < np.pi / 2:
                normal = np.cross(vecs[0], vecs[1])
                check0 = (diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[2]) > 0
                normal = np.cross(vecs[2], vecs[3])
                check1 = (diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[2]) > 0
                normal = np.cross(vecs[1], vecs[2])
                check2 = (diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[2]) > 0
                normal = np.cross(vecs[3], vecs[0])
                check3 = (diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[2]) > 0
                check = (check0 & check1 & check2 & check3).reshape(-1, 1)

                data_output = tuple()
                for array in data:
                    data_output += self.array_slice(array, check, mode)
                return data_output

            elif ang_bw_two_vects(np.cross(vecs[0], vecs[1]), vecs[2]) > np.pi / 2:
                normal = np.cross(vecs[1], vecs[0])
                check0 = (diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[2]) > 0
                normal = np.cross(vecs[3], vecs[2])
                check1 = (diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[2]) > 0
                normal = np.cross(vecs[2], vecs[1])
                check2 = (diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[2]) > 0
                normal = np.cross(vecs[0], vecs[3])
                check3 = (diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[2]) > 0
                check = (check0 & check1 & check2 & check3).reshape(-1, 1)

                data_output = tuple()
                for array in data:
                    data_output += self.array_slice(array, check, mode)
                return data_output

        elif self.geometry == 'circle':
            if self.orientation == 'independent':
                D = -vecs[0] * normal[0] - vecs[1] * normal[1] - vecs[2] * normal[2]
                scalar_sum = (diff_vecs[:, 0] * normal[0]).reshape(-1, 1) + (diff_vecs[:, 1] * normal[1]).reshape(-1,
                                                                                                                  1) + (
                                     diff_vecs[:, 2] * normal[2]).reshape(-1, 1)
                check0 = scalar_sum < 0

                x_intersection = (-D * diff_vecs[:, 0] / (
                        diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[
                    2])).reshape(-1, 1)
                y_intersection = (-D * diff_vecs[:, 1] / (
                        diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[
                    2])).reshape(-1, 1)
                z_intersection = (-D * diff_vecs[:, 2] / (
                        diff_vecs[:, 0] * normal[0] + diff_vecs[:, 1] * normal[1] + diff_vecs[:, 2] * normal[
                    2])).reshape(-1, 1)
                data_intersection = np.hstack((x_intersection, y_intersection, z_intersection)) - vec_origin_to_centre
                check1 = ((data_intersection[:, 0] ** 2 + data_intersection[:, 1] ** 2 + data_intersection[:, 2] ** 2)
                          < (diameter / 2) ** 2).reshape(-1, 1)
                check = check0 & check1
                data_output = tuple()
                for array in data:
                    data_output += self.array_slice(array, check, mode)

                if intersection_coords is True:
                    intersections = np.hstack((x_intersection, y_intersection, z_intersection))
                    intersections = intersections[check1[:, 0] == True].reshape(-1, 3)
                    data_output = data_output + (intersections,)
                return data_output

            elif self.orientation == 'normal':
                max_ang = ang_bw_two_vects(vecs[0], vecs[1])
                angs = ang_bw_two_vects(diff_vecs, vecs[0], type='array')
                check = (angs < max_ang).reshape(-1, 1)
                data_output = tuple()
                for array in data:
                    data_output += self.array_slice(array, check, mode)
                return data_output

    def sift(self,
             data: Tuple[np.ndarray, ...],
             hkl: np.ndarray,
             limit_abs: float = 0.9,
             reject_all_walls: bool = False,
             sift_with_back: bool = False) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:

        max_i = 1 - limit_abs
        data, hkl = ((data), (hkl))
        sifted_data = ()
        sifted_hkl = ()
        if reject_all_walls is True:
            for i in range(len(data)):
                sifted_data = sifted_data + (data[i][data[i][:, 3] == 0],)
                sifted_hkl = sifted_hkl + (hkl[i][data[i][:, 3] == 0],)
            data = sifted_data
            hkl = sifted_hkl
            sifted_data = ()
            sifted_hkl = ()

        if sift_with_back is True:
            for i in range(len(data)):
                sifted_data = sifted_data + (data[i][data[i][:, 4] < max_i],)
                sifted_hkl = sifted_hkl + (hkl[i][data[i][:, 4] < max_i],)
            data = sifted_data
            hkl = sifted_hkl
        elif sift_with_back is False:
            for i in range(len(data)):
                data_back = data[i][data[i][:, 3] == 0]
                hkl_back = hkl[i][data[i][:, 3] == 0]
                data_walls = data[i][data[i][:, 3] != 0]
                hkl_walls = hkl[i][data[i][:, 3] != 0]
                hkl_walls = hkl_walls[data_walls[:, 4] < max_i]
                data_walls = data_walls[data_walls[:, 4] < max_i]
                sifted_data = sifted_data + (np.vstack((data_back, data_walls)),)
                sifted_hkl = sifted_hkl + (np.vstack((hkl_back, hkl_walls)),)
            data = sifted_data
            hkl = sifted_hkl
        return (data, hkl)

    def trash_dead_areas(self, data: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
        if self.dead_areas is not None:
            rotation = R.from_euler('xyz', self.rot, degrees=True)
            for dead_area in self.dead_areas:
                if self.geometry == 'rectangle':
                    width = self.width
                    height = self.width
                else:
                    width = self.diameter
                    height = self.diameter

                vecs = np.array([[0, dead_area[0][0] * width / 2, dead_area[0][1] * height / 2],
                                 [0, dead_area[1][0] * width / 2, dead_area[1][1] * height / 2],
                                 [0, dead_area[2][0] * width / 2, dead_area[2][1] * height / 2],
                                 [0, dead_area[3][0] * width / 2, dead_area[3][1] * height / 2]])
                if self.orientation == 'normal':
                    vecs += self.vec_origin_to_centre
                    vecs = rotation.apply(vecs)
                elif self.orientation == 'independent':
                    vecs = rotation.apply(vecs)
                    vecs += self.vec_origin_to_centre
                data = self.filter(diff_vecs=data[0], data=data, mode='shade', vecs=vecs)
            return data
        else:
            return data


# class LinkedObstacle(Ray_obstacle):
#     def __init__(self,
#                  dist: float,
#                  geometry: str,
#                  disp_y: float,
#                  disp_z: float,
#                  rot: np.ndarray,
#                  orientation: str,
#                  height: Optional[float] = None,
#                  width: Optional[float] = None,
#                  diameter: Optional[float] = None,
#                  symmetric: bool = False,
#                  highest_linked_axis_index: int = 0,
#                  ):
#         self.dist = dist
#         self.geometry = geometry
#         self.disp_y = disp_y
#         self.disp_z = disp_z
#         self.rot = rot
#         self.orientation = orientation
#         self.height = height
#         self.width = width
#         self.diameter = diameter
#         self.symmetric = symmetric
#         self.highest_linked_axis_index = highest_linked_axis_index
#
#     def filter(self,
#                diff_vecs: np.ndarray,
#                angles: np.ndarray,
#                data: Tuple[np.ndarray, ...],
#                mode: str,
#                intersection_coords: bool = False,
#                normal: Optional[np.ndarray] = None,
#                vec_origin_to_centre: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
#         pass
#
#     def create_obst_vec_arr(self,
#                             obst_vecs: NDArray[Shape["1, 3"], Float],
#                             diff_angles: NDArray[Shape["-1, 1"], Float],
#                             rotation_axes: str,
#                             directions_axes: Tuple[int, ...],
#                             initial_axes_angles: Tuple[float, ...],
#                             scan_axis_index: int
#                              ):
#         assert set(rotation_axes).issubset(['x', 'y', 'z']) and rotation_axes != '', 'Wrong rotation axes!'
#         n_obst_vecs = obst_vecs.shape[0]
#         n_angles = len(diff_angles)
#         obst_vecs_ = np.tile(obst_vecs, (n_angles, 1, 1))
#         rot_a = rotation_axes[scan_axis_index]
#         rot_vecs = {'x': np.array([1, 0, 0]),
#                     'y': np.array([0, 1, 0]),
#                     'z': np.array([0, 0, 1])}
#         rot_vec = rot_vecs.get(rot_a)
#         rot_vecs = rot_vec * directions_axes
#         rot_vecs = np.repeat(rot_vecs, repeats=n_obst_vecs, axis=0)
#         rotations = R.from_rotvec(rot_vecs)
#         obst_vecs_ = rotations.apply(obst_vecs_.reshape(-1, 3)).reshape(n_obst_vecs, -1, 3)
#         return obst_vecs_
#
#     def create_obst_vecs(self,):
#         pass


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
        n_reflections = diff_vecs.shape[0]
        angles_deg = np.rad2deg(diff_angles)
        anvil_normals = np.tile(self.normal, (n_reflections, 1))
        incident_beam = incident_beam / np.linalg.norm(incident_beam).reshape(-1)
        matr1, matr3 = Sample.generate_rotation_matrices(rotations=rotation_axes, directions=directions_axes,
                                                         angle=initial_axes_angles, no_of_scan=scan_axis_index)
        rotations = R.from_matrix(matr1) * R.from_euler(rotation_axes[scan_axis_index],
                                                        diff_angles * directions_axes[scan_axis_index]) * R.from_matrix(
            matr3)
        anvil_normals = rotations.apply(anvil_normals)
        cos_bw_incident_norm = (anvil_normals[:, 0] * incident_beam[0] + anvil_normals[:, 1] * incident_beam[1]
                                + anvil_normals[:, 2] * incident_beam[2])
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


class Sample():
    def __init__(self,
                 orient_matx: Optional[np.ndarray] = None,
                 a: Optional[float] = None,
                 b: Optional[float] = None,
                 c: Optional[float] = None,
                 al: Optional[float] = None,
                 bt: Optional[float] = None,
                 gm: Optional[float] = None,
                 om: Optional[float] = None,
                 phi: Optional[float] = None,
                 chi: Optional[float] = None,
                 ):

        if orient_matx is not None:
            self.orient_matx = orient_matx
            prm_rec = np.hstack((np.linalg.norm([orient_matx[0, 0], orient_matx[1, 0], orient_matx[2, 0]]),
                                 np.linalg.norm([orient_matx[0, 1], orient_matx[1, 1], orient_matx[2, 1]]),
                                 np.linalg.norm([orient_matx[0, 2], orient_matx[1, 2], orient_matx[2, 2]]),
                                 self.ang_bw_two_vects(orient_matx[:, 1].reshape(-1), orient_matx[:, 2].reshape(-1)),
                                 self.ang_bw_two_vects(orient_matx[:, 0].reshape(-1), orient_matx[:, 2].reshape(-1)),
                                 self.ang_bw_two_vects(orient_matx[:, 0].reshape(-1), orient_matx[:, 1].reshape(-1))))
            i = np.linalg.inv(self.orient_matx)

            parameters = np.round(np.hstack((np.linalg.norm([i[0, 0], i[0, 1], i[0, 2]]),
                                             np.linalg.norm([i[1, 0], i[1, 1], i[1, 2]]),
                                             np.linalg.norm([i[2, 0], i[2, 1], i[2, 2]]))), 4)
            self.parameters = np.hstack(
                (parameters, self.ang_bw_two_vects(i[1], i[2]), self.ang_bw_two_vects(i[0], i[2]),
                 self.ang_bw_two_vects(i[1], i[0])))
            self.cell_vol = 1 / np.abs(np.linalg.det(orient_matx))
            self.b_matrix = np.array([
                [prm_rec[0], prm_rec[1] * np.cos(np.deg2rad(prm_rec[5])), prm_rec[2] * np.cos(np.deg2rad(prm_rec[4]))],
                [0, prm_rec[1] * np.sin(np.deg2rad(prm_rec[5])),
                 -prm_rec[2] * np.sin(np.deg2rad(prm_rec[4])) * np.cos(np.deg2rad(self.parameters[3]))],
                [0, 0, 1 / self.parameters[2]],
            ])

        elif orient_matx is None:
            self.parameters = np.array([a, b, c, al, bt, gm])
            al = np.deg2rad(al)
            bt = np.deg2rad(bt)
            gm = np.deg2rad(gm)
            if all((phi, chi, om)):
                phi = np.deg2rad(phi)
                chi = np.deg2rad(chi)
                om = np.deg2rad(om)
            else:
                phi = .0
                chi = .0
                om = .0
            cellvolume = self.volume_by_parameters(a, b, c, al, bt, gm)
            self.cell_vol = cellvolume
            arec = 1 / self.create_d_array(self.parameters, cellvolume, np.array([[1, 0, 0]]))[0]
            brec = 1 / self.create_d_array(self.parameters, cellvolume, np.array([[0, 1, 0]]))[0]
            crec = 1 / self.create_d_array(self.parameters, cellvolume, np.array([[0, 0, 1]]))[0]

            btrec = np.arccos((np.cos(al) * np.cos(gm) - np.cos(bt)) / np.sin(al) / np.sin(gm))
            gmrec = np.arccos((np.cos(al) * np.cos(bt) - np.cos(gm)) / np.sin(al) / np.sin(bt))
            self.b_matrix = np.array([[arec, brec * np.cos(gmrec), crec * np.cos(btrec)],
                                      [0, brec * np.sin(gmrec), -crec * np.sin(btrec) * np.cos(al)],
                                      [0, 0, 1 / c]])
            self.orient_matx = R.from_euler('zxz', (om, -chi, -phi)).apply(self.b_matrix)

    def volume_by_parameters(self, a: float, b: float, c: float, al: float, bt: float, gm: float) -> float:
        cellvolume = a * b * c * (1 - np.cos(al) ** 2 - np.cos(bt) ** 2 - np.cos(gm) ** 2 + 2 * np.cos(al) * np.cos(bt)
                                  * np.cos(gm)) ** 0.5
        return cellvolume

    def ang_bw_two_vects(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_prod = np.dot(vec1, vec2)
        angle = np.arccos(dot_prod / np.linalg.norm(vec1) / np.linalg.norm(vec2))
        return np.rad2deg(angle)

    @staticmethod
    def generate_rotation_matrices(
            rotations: str,
            directions: Tuple[int, ...],
            angle: Tuple[float, ...],
            no_of_scan: int = 1) -> Tuple[np.ndarray, np.ndarray]:

        angle = np.array(angle) * np.array(directions)
        angle = np.deg2rad(angle)

        num_of_rots = len(rotations)
        if num_of_rots == 1:
            matr1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            matr3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            return (matr1, matr3)
        else:
            if no_of_scan == 0:
                matr1 = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                rotations1 = rotations[1:][::-1]
                angle1 = angle[1:][::-1]
                for i in range(len(rotations1)):
                    matr1 = matr1 * R.from_euler(rotations1[i], angle1[i])

                matr3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                return (matr1.as_matrix(), matr3)

            elif no_of_scan + 1 == num_of_rots:
                matr1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                matr3 = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                rotations3 = rotations[:-1][::-1]
                angle3 = angle[:-1][::-1]
                for i in range(len(rotations3)):
                    matr3 = matr3 * R.from_euler(rotations3[i], angle3[i])
                return (matr1, matr3.as_matrix())

            elif no_of_scan != 0 and no_of_scan + 1 != num_of_rots:
                matr1 = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                rotations1 = rotations[no_of_scan+1:][::-1]
                angle1 = angle[no_of_scan+1:][::-1]
                for i in range(len(rotations1)):
                    matr1 = matr1 * R.from_euler(rotations1[i], angle1[i])

                matr3 = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                rotations3 = rotations[:no_of_scan][::-1]
                angle3 = angle[:no_of_scan][::-1]
                for i in range(len(rotations3)):
                    matr3 = matr3 * R.from_euler(rotations3[i], angle3[i])

                return (matr1.as_matrix(), matr3.as_matrix())

    def coefficient_z(self,
                      matr13: Tuple[np.ndarray, np.ndarray],
                      vectors: np.ndarray,
                      scan: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if scan == 'z':
            k1 = vectors[:, 0] * matr13[0][0, 2] * matr13[1][2, 0]
            k2 = vectors[:, 0] * matr13[0][0, 0] * matr13[1][0, 0]
            k3 = vectors[:, 0] * matr13[0][0, 1] * matr13[1][0, 0]
            k4 = vectors[:, 0] * matr13[0][0, 0] * matr13[1][1, 0]
            k5 = vectors[:, 0] * matr13[0][0, 1] * matr13[1][1, 0]

            k6 = vectors[:, 1] * matr13[0][0, 2] * matr13[1][2, 1]
            k7 = vectors[:, 1] * matr13[0][0, 0] * matr13[1][0, 1]
            k8 = vectors[:, 1] * matr13[0][0, 1] * matr13[1][0, 1]
            k9 = vectors[:, 1] * matr13[0][0, 0] * matr13[1][1, 1]
            k10 = vectors[:, 1] * matr13[0][0, 1] * matr13[1][1, 1]

            k11 = vectors[:, 2] * matr13[0][0, 2] * matr13[1][2, 2]
            k12 = vectors[:, 2] * matr13[0][0, 0] * matr13[1][0, 2]
            k13 = vectors[:, 2] * matr13[0][0, 1] * matr13[1][0, 2]
            k14 = vectors[:, 2] * matr13[0][0, 0] * matr13[1][1, 2]
            k15 = vectors[:, 2] * matr13[0][0, 1] * matr13[1][1, 2]

            l1 = k1 + k6 + k11
            l2 = k2 + k5 + k7 + k10 + k12 + k15
            l3 = k3 - k4 + k8 - k9 + k13 - k14
            return (l1, l2, l3)

        elif scan == 'x':
            k1 = vectors[:, 0] * matr13[0][0, 0] * matr13[1][0, 0]
            k2 = vectors[:, 0] * matr13[0][0, 1] * matr13[1][1, 0]
            k3 = vectors[:, 0] * matr13[0][0, 2] * matr13[1][1, 0]
            k4 = vectors[:, 0] * matr13[0][0, 1] * matr13[1][2, 0]
            k5 = vectors[:, 0] * matr13[0][0, 2] * matr13[1][2, 0]

            k6 = vectors[:, 1] * matr13[0][0, 0] * matr13[1][0, 1]
            k7 = vectors[:, 1] * matr13[0][0, 1] * matr13[1][1, 1]
            k8 = vectors[:, 1] * matr13[0][0, 2] * matr13[1][1, 1]
            k9 = vectors[:, 1] * matr13[0][0, 1] * matr13[1][2, 1]
            k10 = vectors[:, 1] * matr13[0][0, 2] * matr13[1][2, 1]

            k11 = vectors[:, 2] * matr13[0][0, 0] * matr13[1][0, 2]
            k12 = vectors[:, 2] * matr13[0][0, 1] * matr13[1][1, 2]
            k13 = vectors[:, 2] * matr13[0][0, 2] * matr13[1][1, 2]
            k14 = vectors[:, 2] * matr13[0][0, 1] * matr13[1][2, 2]
            k15 = vectors[:, 2] * matr13[0][0, 2] * matr13[1][2, 2]

            l1 = k1 + k6 + k11
            l2 = k2 + k5 + k7 + k10 + k12 + k15
            l3 = k3 - k4 + k8 - k9 + k13 - k14
            return (l1, l2, l3)

        elif scan == 'y':
            k1 = vectors[:, 0] * matr13[0][0, 1] * matr13[1][1, 0]
            k2 = vectors[:, 0] * matr13[0][0, 0] * matr13[1][0, 0]
            k3 = vectors[:, 0] * matr13[0][0, 0] * matr13[1][2, 0]
            k4 = vectors[:, 0] * matr13[0][0, 2] * matr13[1][0, 0]
            k5 = vectors[:, 0] * matr13[0][0, 2] * matr13[1][2, 0]

            k6 = vectors[:, 1] * matr13[0][0, 1] * matr13[1][1, 1]
            k7 = vectors[:, 1] * matr13[0][0, 0] * matr13[1][0, 1]
            k8 = vectors[:, 1] * matr13[0][0, 0] * matr13[1][2, 1]
            k9 = vectors[:, 1] * matr13[0][0, 2] * matr13[1][0, 1]
            k10 = vectors[:, 1] * matr13[0][0, 2] * matr13[1][2, 1]

            k11 = vectors[:, 2] * matr13[0][0, 1] * matr13[1][1, 2]
            k12 = vectors[:, 2] * matr13[0][0, 0] * matr13[1][0, 2]
            k13 = vectors[:, 2] * matr13[0][0, 0] * matr13[1][2, 2]
            k14 = vectors[:, 2] * matr13[0][0, 2] * matr13[1][0, 2]
            k15 = vectors[:, 2] * matr13[0][0, 2] * matr13[1][2, 2]

            l1 = k1 + k6 + k11
            l2 = k2 + k5 + k7 + k10 + k12 + k15
            l3 = k3 - k4 + k8 - k9 + k13 - k14
            return (l1, l2, l3)

    def run(self,
            rotations: str,
            directions: Tuple[int, int, int],
            angle: Tuple[float, float, float],
            no_of_scan: int,
            vectors: np.ndarray,
            cos_si_ki: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        matr13 = self.generate_rotation_matrices(rotations, directions, angle, no_of_scan)
        c = self.coefficient_z(matr13, vectors, scan=rotations[no_of_scan])
        if directions[no_of_scan] == 1:
            angle1 = -2 * np.arctan(
                (c[2] - np.sqrt(-c[0] ** 2 - 2 * c[0] * cos_si_ki + c[1] ** 2 + c[2] ** 2 - cos_si_ki ** 2)) / (
                        c[0] - c[1] + cos_si_ki))
            angle2 = -2 * np.arctan(
                (c[2] + np.sqrt(-c[0] ** 2 - 2 * c[0] * cos_si_ki + c[1] ** 2 + c[2] ** 2 - cos_si_ki ** 2)) / (
                        c[0] - c[1] + cos_si_ki))
        elif directions[no_of_scan] == -1:
            angle1 = 2 * np.arctan(
                (c[2] - np.sqrt(-c[0] ** 2 - 2 * c[0] * cos_si_ki + c[1] ** 2 + c[2] ** 2 - cos_si_ki ** 2)) / (
                        c[0] - c[1] + cos_si_ki))
            angle2 = 2 * np.arctan(
                (c[2] + np.sqrt(-c[0] ** 2 - 2 * c[0] * cos_si_ki + c[1] ** 2 + c[2] ** 2 - cos_si_ki ** 2)) / (
                        c[0] - c[1] + cos_si_ki))
        return (angle1, angle2, matr13)

    @staticmethod
    @mylogger('DEBUG')
    def create_d_array(parameters: np.ndarray,
                       cell_vol: float,
                       hkl_array: np.ndarray) -> np.ndarray:

        a = parameters[0]
        b = parameters[1]
        c = parameters[2]
        al = np.deg2rad(parameters[3])
        bt = np.deg2rad(parameters[4])
        gm = np.deg2rad(parameters[5])
        d_array = cell_vol / (
                hkl_array[:, 0] ** 2 * b ** 2 * c ** 2 * np.sin(al) ** 2 + hkl_array[:,
                                                                           1] ** 2 * a ** 2 * c ** 2 * np.sin(
            bt) ** 2 + hkl_array[:, 2] ** 2 * a ** 2 * b ** 2 * np.sin(gm) ** 2 +
                2 * hkl_array[:, 0] * hkl_array[:, 1] * a * b * c ** 2 * (np.cos(al) * np.cos(bt) - np.cos(gm)) + 2 *
                hkl_array[:, 2] * hkl_array[:, 1] * a ** 2 * b * c * (np.cos(gm) * np.cos(bt) - np.cos(al)) + 2 *
                hkl_array[:, 0] * hkl_array[:, 2] * a * b ** 2 * c * (np.cos(al) * np.cos(gm) - np.cos(bt))) ** 0.5
        return d_array
    @staticmethod
    def angle_range(
                    scan_sweep: float,
                    start_angle: float,
                    epsilon: float = 1e-12,
                    start_rad:Union[None,float] = None,
                    end_rad:Union[None,float] = None
                    ) -> Tuple[float, float, str]:
        if not start_rad and not end_rad:
            start_rad = np.deg2rad(start_angle) % (2 * np.pi)
            sweep_rad = np.deg2rad(scan_sweep)

            end_rad = start_rad + sweep_rad

        if 0 - epsilon <= end_rad <= 2 * np.pi + epsilon:
            return (min(start_rad, end_rad), max(start_rad, end_rad), 'in')

        end_norm = end_rad % (2 * np.pi)
        start_norm = start_rad % (2 * np.pi)
        return (min(start_norm, end_norm), max(start_norm, end_norm), 'ex')

    def angles_in_sweep(self,
                        angles_array: np.ndarray,
                        start: float,
                        end: float,
                        type: str) -> np.ndarray:

        if end - start > np.pi * 2:
            return angles_array
        else:
            if type == 'in':
                angles_array[angles_array < start] = np.nan
                angles_array[angles_array > end] = np.nan
                return angles_array
            elif type == 'ex':
                i = angles_array.copy()
                i[angles_array < start] = np.nan
                i[angles_array > end] = np.nan
                angles_array[~np.isnan(i)] = np.nan
                return angles_array

    @mylogger(level='DEBUG', log_args=True, log_result=True)
    def gen_hkl_arrays(self,
                       type: str,
                       d_range: Optional[Tuple[float, float]] = None,
                       hkl_section: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
                       return_origin: bool = False,
                       pg: Optional[str] = None,
                       centring: str = 'P') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if d_range is not None and hkl_section is not None:
            warnings.warn(
                'Note, Both d, and hkl parameters are not None. Only one parameter will be used according to the type parameter')
        if return_origin is True and type == 'hkl_section':
            warnings.warn('return_origin is True and type is hkl_section, which is meaningless')

        if type == 'hkl_section':
            h_array = np.arange(hkl_section[0][0], hkl_section[0][1] + 1, 1)
            k_array = np.arange(hkl_section[1][0], hkl_section[1][1] + 1, 1)
            l_array = np.arange(hkl_section[2][0], hkl_section[2][1] + 1, 1)
            hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
            return hkl_array

        elif type == 'd_range':
            pg_key = PG_KEYS[pg]
            a, b, c, al, bt, gm = self.parameters
            cell_vol = self.cell_vol
            al_rec = self.ang_bw_two_vects(self.orient_matx[:, 1].reshape(-1), self.orient_matx[:, 2].reshape(-1))
            bt_rec = self.ang_bw_two_vects(self.orient_matx[:, 0].reshape(-1), self.orient_matx[:, 2].reshape(-1))
            gm_rec = self.ang_bw_two_vects(self.orient_matx[:, 0].reshape(-1), self.orient_matx[:, 1].reshape(-1))
            bc_low_d = d_range[0] / np.sin(np.deg2rad(al_rec))
            ac_low_d = d_range[0] / np.sin(np.deg2rad(bt_rec))
            ab_low_d = d_range[0] / np.sin(np.deg2rad(gm_rec))
            h1_max = abs(int(cell_vol / ac_low_d / b / c / np.sin(np.deg2rad(al)))) * 2
            h2_max = abs(int(cell_vol / ab_low_d / b / c / np.sin(np.deg2rad(al)))) * 2
            k1_max = abs(int(cell_vol / bc_low_d / a / c / np.sin(np.deg2rad(bt)))) * 2
            k2_max = abs(int(cell_vol / ab_low_d / a / c / np.sin(np.deg2rad(bt)))) * 2
            l1_max = abs(int(cell_vol / bc_low_d / b / a / np.sin(np.deg2rad(gm)))) * 2
            l2_max = abs(int(cell_vol / ac_low_d / b / a / np.sin(np.deg2rad(gm)))) * 2
            hmax = h1_max if h1_max > h2_max else h2_max
            kmax = k1_max if k1_max > k2_max else k2_max
            lmax = l1_max if l1_max > l2_max else l2_max

            hkl_orig_array = generate_orig_hkl_array(hmax, kmax, lmax, pg_key, centring)
            d_array = self.create_d_array(self.parameters, self.cell_vol, hkl_orig_array).reshape(-1, 1)
            d_range = np.array(np.float32(d_range))
            d_range[0] -= 0.0000001
            d_range[1] += 0.0000001
            hkl_orig_array = hkl_orig_array[d_array[:, 0] > d_range[0]]

            d_array = d_array[d_array > d_range[0]].reshape(-1, 1)
            hkl_orig_array = hkl_orig_array[d_array[:, 0] < d_range[1]]

            hkl_array, original_hkl, hkl_orig = generate_hkl_by_pg(hkl_orig_array, pg_key)

            if return_origin is False:
                return hkl_array
            elif return_origin is True:
                return hkl_array, original_hkl

    def scan(self,
             scan_type: str = '???',
             scan_sweep: float = 360,
             rotations: str = 'zxz',
             angles: Tuple[float, float, float] = (1., 2., 3.),
             directions: Tuple[int, int, int] = (-1, -1, 1),
             no_of_scan: int = 1,
             hkl_array: Optional[np.ndarray] = None,
             hkl_array_orig: Optional[np.ndarray] = None,
             wavelength: float = 0.71073,
             only_angles: bool = False) -> Union[Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

        if hkl_array is None:
            raise Exception('hkl_array is None, please provide some reflections as numpy array object (-1,3)')
        elif hkl_array_orig is None:
            hkl_array_orig = hkl_array

        d_array = self.create_d_array(self.parameters, self.cell_vol, hkl_array).reshape(-1, 1)
        cos_s_ki_ang_array = np.cos(np.deg2rad(90) - np.arcsin(wavelength / 2 / d_array))
        hkl_rotated = np.matmul(self.orient_matx, hkl_array.reshape(-1, 3, 1)).reshape(-1, 3)
        hkl_len = ((hkl_rotated[:, 0] ** 2 + hkl_rotated[:, 1] ** 2 + hkl_rotated[:, 2] ** 2) ** 0.5).reshape(-1, 1)
        hkl_array_norm = (hkl_rotated / hkl_len)
        hkl_array_norm = np.hstack((hkl_array_norm, cos_s_ki_ang_array))

        if scan_type == '???':
            angle_start = angles[no_of_scan]
            range = self.angle_range(scan_sweep, angle_start)
            hkl_array = hkl_array.reshape(-1, 3)
            T = hkl_array_norm[:, -1]
            hkl_array_norm = hkl_array_norm[:, :3]
            solution = self.run(rotations, directions, angles, no_of_scan, hkl_array_norm.copy(), T)

            angle1 = solution[0].reshape(-1, 1)
            angle2 = solution[1].reshape(-1, 1)
            matr1 = solution[2][0]
            matr3 = solution[2][1]

            angle1[angle1 < 0] = 2 * np.pi + angle1[angle1 < 0]
            angle2[angle2 < 0] = 2 * np.pi + angle2[angle2 < 0]

            angle1 = self.angles_in_sweep(angle1, range[0], range[1], range[2])
            angle2 = self.angles_in_sweep(angle2, range[0], range[1], range[2])
            if only_angles is True:
                return (angle1, angle2)

            hkl_rotated1 = hkl_rotated.copy()[~np.isnan(angle1[:, 0])].reshape(-1, 3)
            hkl_rotated2 = hkl_rotated.copy()[~np.isnan(angle2[:, 0])].reshape(-1, 3)
            hkl_array1 = hkl_array.copy()[~np.isnan(angle1[:, 0])].reshape(-1, 3)
            hkl_array2 = hkl_array.copy()[~np.isnan(angle2[:, 0])].reshape(-1, 3)
            hkl_array_orig1 = hkl_array_orig.copy()[~np.isnan(angle1[:, 0])].reshape(-1, 3)
            hkl_array_orig2 = hkl_array_orig.copy()[~np.isnan(angle2[:, 0])].reshape(-1, 3)
            angle1 = angle1[~np.isnan(angle1)].reshape(-1, 1)
            angle2 = angle2[~np.isnan(angle2)].reshape(-1, 1)

            if angle1.shape != (0,1):
                rotation1 = R.from_matrix(matr1) * R.from_euler(rotations[no_of_scan],
                                                                angle1 * directions[no_of_scan]) * R.from_matrix(matr3)
                hkl_rotated1 = rotation1.apply(hkl_rotated1)
                diff_vec1 = hkl_rotated1 + np.array([1 / wavelength, 0, 0])
            else:
                diff_vec1 = np.array([]).reshape(-1,3)

            if angle2.shape != (0,1):
                rotation2 = R.from_matrix(matr1) * R.from_euler(rotations[no_of_scan],
                                                                angle2 * directions[no_of_scan]) * R.from_matrix(matr3)
                hkl_rotated2 = rotation2.apply(hkl_rotated2)
                diff_vec2 = hkl_rotated2 + np.array([1 / wavelength, 0, 0])
            else:
                diff_vec2 = np.array([]).reshape(-1,3)



            angles_all = np.vstack((angle1, angle2))

            hkl_all = np.vstack((hkl_array1, hkl_array2))
            hkl_array_orig_all = np.vstack((hkl_array_orig1, hkl_array_orig2))
            diff_vec_all = np.vstack((diff_vec1, diff_vec2))
            diff_vec_all = diff_vec_all / np.linalg.norm(diff_vec_all, axis=1).reshape(-1, 1)

            return (diff_vec_all, hkl_all, hkl_array_orig_all, angles_all)

    def calc_scan_vect(self,
                       rotations: str,
                       angles: Tuple[float, float, float],
                       directions: Tuple[int, int, int],
                       n_of_scan: int) -> np.ndarray:

        if directions[n_of_scan] == 'x':
            scan_axis = np.array([1, 0, 0])
        elif directions[n_of_scan] == 'y':
            scan_axis = np.array([0, 1, 0])
        else:
            scan_axis = np.array([0, 0, 1])

        vec_rotations = rotations[:n_of_scan][::-1]
        angles = list(np.array(angles) * np.array(directions))
        angles_rotation = angles[:n_of_scan][::-1]
        if len(vec_rotations) == 0:
            pass
        else:
            rot_matr = R.from_matrix(np.eye(3))
            for i, rot in enumerate(vec_rotations):
                rotation = R.from_euler(rot, -angles_rotation[i], degrees=True)
                rot_matr = rotation * rot_matr
            scan_axis = rot_matr.apply(scan_axis)
        return scan_axis

    def zenith_azimuth_rev_vect(self,
                                vector: np.ndarray,
                                points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        zenith = np.rad2deg(ang_bw_two_vects(points, vector, type='array'))
        t = (vector[0] * points[:, 0] + vector[1] * points[:, 1] + vector[2] * points[:, 2]) / \
            (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
        points_projections = points.astype(float)
        points_projections[:, 0] -= t * vector[0]
        points_projections[:, 1] -= t * vector[1]
        points_projections[:, 2] -= t * vector[2]
        rotation = R.align_vectors(vector.reshape(1, 3), np.array([[0, 0, 1]]))
        points_projections = rotation[0].apply(points_projections)
        azimuth = np.rad2deg(np.arctan(points_projections[:, 1] / points_projections[:, 0]))
        return zenith, azimuth

    def mapv2(self,
              reflections: np.ndarray,
              rotations: str,
              directions: Tuple[int, int, int],
              angles: Tuple[float, float, float],
              rotation_directions: Tuple[int, int, int],
              steps: Tuple[float, float],
              ranges: Tuple[Tuple[float, float], Tuple[float, float]],
              wavelength: float,
              names: Tuple[str, str, str],
              visualise: bool = True) -> Union[None, go.Figure]:

        x_range = np.arange(ranges[0][0], ranges[0][1], steps[1])
        z_range = np.arange(ranges[1][0], ranges[1][1], steps[1])
        x_len = len(x_range)
        z_len = len(z_range)
        angles_ = angles
        n_of_reflections = reflections.shape[0]
        first = True
        for anglez in z_range:
            angles_[rotation_directions[2]] = anglez
            for anglex in x_range:
                angles_[rotation_directions[1]] = anglex

                angle1, angle2 = self.scan(scan_type='???', scan_sweep=360, rotations=rotations, angles=angles_,
                                           directions=directions, no_of_scan=rotation_directions[0],
                                           hkl_array=reflections, hkl_array_orig=reflections,
                                           wavelength=wavelength, only_angles=True)
                if first is True:
                    first = False
                    angle1_array = angle1
                    angle2_array = angle2
                else:
                    angle1_array = np.concatenate((angle1_array, angle1))
                    angle2_array = np.concatenate((angle2_array, angle2))
        angle1_array = np.rad2deg(np.vstack((angle1_array, angle2_array)))
        x_angle_array = np.tile(np.tile(np.tile(np.array(x_range).reshape(-1, 1), (1, n_of_reflections)).reshape(-1, 1),
                                        (z_len, 1)), (2, 1))
        z_angle_array = np.tile(np.tile(np.array(z_range).reshape(-1, 1), (1, n_of_reflections * x_len)).reshape(-1, 1),
                                (2, 1))

        hkl_str = np.tile(np.tile(
            np.char.add(np.char.add(np.char.add(np.char.add(reflections[:, 0].astype(int).astype(str), ' '),
                                                reflections[:, 1].astype(int).astype(str)), ' '),
                        reflections[:, 2].astype(int).astype(str)).reshape(-1, 1), (x_len * z_len, 1)), (2, 1))

        random_rgb_colors1 = (
            np.round(np.tile(np.tile(np.random.rand(n_of_reflections), (1, x_len * z_len)), (1, 2)) * 360)).astype(str)
        random_rgb_colors2 = (
            np.round(np.tile(np.tile(np.random.rand(n_of_reflections), (1, x_len * z_len)), (1, 2)) * 360)).astype(str)
        random_rgb_colors3 = (
            np.round(np.tile(np.tile(np.random.rand(n_of_reflections), (1, x_len * z_len)), (1, 2)) * 360)).astype(str)
        color_rgb_array = np.char.add(np.char.add(
            np.char.add(np.char.add(np.char.add(np.char.add('rgb(', random_rgb_colors1), ', '), random_rgb_colors2, ),
                        ', '), random_rgb_colors3), ')')

        data = {
            f'{names[0]}': angle1_array.reshape(-1),
            f'{names[1]}': x_angle_array.reshape(-1),
            f'{names[2]}': z_angle_array.reshape(-1),
            'hkl': hkl_str.reshape(-1),
        }
        df = pd.DataFrame(data=data)
        fig = px.scatter_3d(df, x=f'{names[0]}', y=f'{names[1]}', z=f'{names[2]}', color='hkl')
        fig.update_traces(marker_size=1.5)
        fig.layout.scene.camera.projection.type = "orthographic"
        if visualise == True:
            fig.show()
        else:
            return fig

    def map_2d(self,
               reflections: np.ndarray,
               rotations: str,
               directions: Tuple[int, int, int],
               angles: Tuple[float, float, float],
               rotation_directions: Tuple[int, int, int],
               step: float,
               range_x: Tuple[float, float],
               wavelength: float,
               names: Tuple[str, str],
               visualise: bool = True) -> Union[None, go.Figure]:

        x_range = np.arange(range_x[0], range_x[1], step).astype(float)
        x_len = len(x_range)
        angles_ = angles
        n_of_reflections = reflections.shape[0]
        first = True
        for anglex in x_range:
            angles_[rotation_directions[1]] = anglex
            angle1, angle2 = self.scan(scan_type='???', scan_sweep=360, rotations=rotations, angles=angles_,
                                       directions=directions, no_of_scan=rotation_directions[0], hkl_array=reflections,
                                       hkl_array_orig=reflections, wavelength=wavelength, only_angles=True)
            if first is True:
                first = False
                angle1_array = angle1
                angle2_array = angle2
            else:
                angle1_array = np.concatenate((angle1_array, angle1))
                angle2_array = np.concatenate((angle2_array, angle2))
        angle_array = np.rad2deg(np.vstack((angle1_array, angle2_array)))
        x_angle_array = np.tile(np.tile(x_range.reshape(-1, 1), (1, n_of_reflections)).reshape(-1, 1), (2, 1))
        x_angle_array[np.isnan(angle_array)] = np.nan
        histx = x_angle_array[~np.isnan(x_angle_array)]
        hkl_str = np.tile(
            np.tile(np.char.add(np.char.add(np.char.add(np.char.add(reflections[:, 0].astype(int).astype(str), ' '),
                                                        reflections[:, 1].astype(int).astype(str)), ' '),
                                reflections[:, 2].astype(int).astype(str)).reshape(-1, 1), (x_len, 1)), (2, 1))
        sort_arr = np.argsort(hkl_str.reshape(2, -1)[0].reshape(-1))
        hkl_str_1 = hkl_str.reshape(2, -1)[0][sort_arr].reshape(n_of_reflections, -1, )
        ang1_sorted = np.rad2deg(angle1_array.reshape(-1)[sort_arr]).reshape(n_of_reflections, -1, )
        ang2_sorted = np.rad2deg(angle2_array.reshape(-1)[sort_arr]).reshape(n_of_reflections, -1, )
        histy = np.array([])
        for n, hkl in enumerate(hkl_str_1):
            histy_new1 = np.unique(np.round(ang1_sorted[n]))
            histy_new2 = np.unique(np.round(ang2_sorted[n]))
            histy = np.concatenate((histy, histy_new1, histy_new2))
        histy = histy[~np.isnan(histy)]
        histy[histy == 360.] = 0.
        random_rgb_colors1 = (
            np.round(np.tile(np.tile(np.random.rand(n_of_reflections), (1, x_len)), (1, 2)) * 255)).astype(int).astype(
            str)
        random_rgb_colors2 = (
            np.round(np.tile(np.tile(np.random.rand(n_of_reflections), (1, x_len)), (1, 2)) * 255)).astype(int).astype(
            str)
        random_rgb_colors3 = (
            np.round(np.tile(np.tile(np.random.rand(n_of_reflections), (1, x_len)), (1, 2)) * 255)).astype(int).astype(
            str)
        color_rgb_array = np.char.add(np.char.add(
            np.char.add(np.char.add(np.char.add(np.char.add('rgb(', random_rgb_colors1), ', '), random_rgb_colors2, ),
                        ', '), random_rgb_colors3), ')')

        data = {
            f'{names[0]}': angle_array.reshape(-1),
            f'{names[1]}': x_angle_array.reshape(-1),
            'colors': color_rgb_array.reshape(-1),
            'hkl': hkl_str
        }

        print(int(np.abs(range_x[0] - range_x[1]) / step))
        fig1 = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True)
        fig1.add_trace(go.Histogram(x=histx, nbinsx=int(np.abs(range_x[0] - range_x[1]) / step)), row=2, col=1)
        fig1.add_trace(go.Histogram(y=histy, nbinsy=360), row=1, col=2)
        fig1.add_trace(
            go.Scatter(x=data[f'{names[1]}'], y=data[f'{names[0]}'], marker={'size': 1.5, 'color': data['colors']},
                       mode='markers', customdata=data['hkl'],
                       hovertemplate='<b>hkl</b>: %{customdata[0]}<b>'),
            row=1, col=1)
        fig1.update_layout({'xaxis_title': f'{names[1]}', 'yaxis_title': f'{names[0]}'}, )

        if visualise == True:
            fig1.show()
        else:
            return fig1

    def map_1d(self,
               reflections: np.ndarray,
               original_hkl: np.ndarray,
               rotations: str,
               directions: Tuple[int, int, int],
               angles: Tuple[float, float, float],
               rotation_direction: int,
               wavelength: float,
               name: str,
               visualise: bool = True) -> Union[None, go.Figure]:

        angle1, angle2 = self.scan(scan_type='???', scan_sweep=360, rotations=rotations, angles=angles,
                                   directions=directions, no_of_scan=rotation_direction, hkl_array=reflections,
                                   hkl_array_orig=reflections, wavelength=wavelength, only_angles=True)
        angle_array = np.round(np.rad2deg(np.vstack((angle1, angle2))))
        angle_array[angle_array == 360.] = 0
        reflections = np.vstack((reflections, reflections))
        original_hkl = np.vstack((original_hkl, original_hkl))
        angle_array = angle_array[~np.isnan(angle_array)]
        fig = px.histogram(x=angle_array, nbins=360, range_x=(0, 359))
        fig['data'][0]['customdata'] = np.hstack((reflections, original_hkl))
        print(fig['data'][0]['customdata'])
        fig.update_layout({'xaxis_title': f'{name}'})
        fig.update_yaxes(fixedrange=True)
        fig.update_xaxes(range=[0, 360])

        if visualise == True:
            fig.show()
        else:
            return fig
