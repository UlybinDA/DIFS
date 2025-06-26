import numpy as np
from scipy.spatial.transform import Rotation as R
from io import StringIO
import warnings
from logger.my_logger import mylogger
from typing import Tuple, Union, Optional, List, Dict, Any
from services.angle_calc import ang_bw_two_vects


warnings.filterwarnings("ignore", category=RuntimeWarning)





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
                 dead_areas: Optional[List[Tuple[float, float]]] = None,
                 name: str = ''):

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
        self.name = name

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
        return None

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
        return None

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
        return None

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

    def _prepare_obstacle_vecs(self, orientation, geometry, rotation, vec_origin_to_centre, disp_y=0, disp_z=0,
                               height=None, width=None, diameter=None):
        if orientation == 'independent':
            if geometry == 'rectangle':
                vecs = np.array(
                    [[0, width / 2, height / 2], [0, width / 2, -height / 2], [0, -width / 2, -height / 2],
                     [0, -width / 2, height / 2]])
                vecs = rotation.apply(vecs)
                vecs = vecs + vec_origin_to_centre

            else:
                vecs = rotation.apply(np.array([[0, diameter / 2, 0]])) + vec_origin_to_centre
                normal = rotation.apply(np.array([[1, 0, 0]]))
                vecs = np.vstack((vecs, normal))



        elif orientation == 'normal':
            if disp_y or disp_z != 0:
                print('disp y or z is not 0 if its made intentionally switch to independent orientation')

            if geometry == 'rectangle':
                vecs = np.array(
                    [[0, width / 2, height / 2], [0, width / 2, -height / 2], [0, -width / 2, -height / 2],
                     [0, -width / 2, height / 2]])
                vecs = vecs + vec_origin_to_centre
                vecs = rotation.apply(vecs)
            else:
                vecs = np.array([[0, 0, 0], [0, 0, diameter / 2]]) + vec_origin_to_centre
                vecs = rotation.apply(vecs)

        return vecs

    def _slice_data(self, data, check, mode):
        data_output = tuple()
        for array in data:
            data_output += self.array_slice(array, check, mode)
        return data_output

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
            vecs = self._prepare_obstacle_vecs(orientation=self.orientation, geometry=self.geometry,
                                               rotation=rotation,
                                               vec_origin_to_centre=self.vec_origin_to_centre,
                                               disp_z=self.disp_z, disp_y=self.disp_y, height=height,
                                               width=width, diameter=diameter)
        else:
            if normal:
                vecs = np.vstack((vecs.reshape(-1, 3), normal.reshape(-1, 3)))

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

                data_output = self._slice_data(data, check, mode)
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

                data_output = self._slice_data(data, check, mode)
                # data_output = tuple()
                # for array in data:
                #     data_output += self.array_slice(array, check, mode)
                return data_output
            return None

        elif self.geometry == 'circle':
            if self.orientation == 'independent':
                D = -vecs[0, 0] * vecs[1, 0] - vecs[0, 1] * vecs[1, 1] - vecs[0, 2] * vecs[1, 2]
                scalar_sum = (diff_vecs[:, 0] * vecs[1, 0]).reshape(-1, 1) + (diff_vecs[:, 1] * vecs[1, 1]).reshape(-1,
                                                                                                                    1) + (
                                     diff_vecs[:, 2] * vecs[1, 2]).reshape(-1, 1)
                check0 = scalar_sum < 0

                x_intersection = (-D * diff_vecs[:, 0] / (
                        diff_vecs[:, 0] * vecs[1, 0] + diff_vecs[:, 1] * vecs[1, 1] + diff_vecs[:, 2] * vecs[1,
                2])).reshape(-1, 1)
                y_intersection = (-D * diff_vecs[:, 1] / (
                        diff_vecs[:, 0] * vecs[1, 0] + diff_vecs[:, 1] * vecs[1, 1] + diff_vecs[:, 2] * vecs[1,
                2])).reshape(-1, 1)
                z_intersection = (-D * diff_vecs[:, 2] / (
                        diff_vecs[:, 0] * vecs[1, 0] + diff_vecs[:, 1] * vecs[1, 1] + diff_vecs[:, 2] * vecs[1,
                2])).reshape(-1, 1)
                data_intersection = np.hstack((x_intersection, y_intersection, z_intersection)) - vec_origin_to_centre
                check1 = ((data_intersection[:, 0] ** 2 + data_intersection[:, 1] ** 2 + data_intersection[:, 2] ** 2)
                          < (diameter / 2) ** 2).reshape(-1, 1)
                check = check0 & check1
                data_output = self._slice_data(data, check, mode)

                if intersection_coords is True:
                    intersections = np.hstack((x_intersection, y_intersection, z_intersection))
                    intersections = intersections[check1[:, 0] == True].reshape(-1, 3)
                    data_output = data_output + (intersections,)
                return data_output

            elif self.orientation == 'normal':
                max_ang = ang_bw_two_vects(vecs[0], vecs[1])
                angs = ang_bw_two_vects(diff_vecs, vecs[0], type='array')
                check = (angs < max_ang).reshape(-1, 1)
                data_output = self._slice_data(data, check, mode)
                return data_output
            return None

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




