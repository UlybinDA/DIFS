import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Union, Optional, List, Dict, Any
from symmetry.pointsymmetry import generate_hkl_by_pg, PG_KEYS, get_key, generate_orig_hkl_array
from services.rotation_wrapper import apply_rotation_vecs
from services.lorentz_wrapper import calc_lorentz
from logger.my_logger import mylogger
import warnings
from services.angle_calc import ang_bw_two_vects

warnings.filterwarnings("ignore", category=RuntimeWarning)


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
            matr1 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            matr3 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            return (matr1, matr3)
        else:
            if no_of_scan == 0:
                matr1 = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                rotations1 = rotations[1:][::-1]
                angle1 = angle[1:][::-1]
                for i in range(len(rotations1)):
                    matr1 = matr1 * R.from_euler(rotations1[i], angle1[i])

                matr3 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                return (matr1.as_matrix(), matr3)

            elif no_of_scan + 1 == num_of_rots:
                matr1 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                matr3 = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                rotations3 = rotations[:-1][::-1]
                angle3 = angle[:-1][::-1]
                for i in range(len(rotations3)):
                    matr3 = matr3 * R.from_euler(rotations3[i], angle3[i])
                return (matr1, matr3.as_matrix())

            elif no_of_scan != 0 and no_of_scan + 1 != num_of_rots:
                matr1 = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                rotations1 = rotations[no_of_scan + 1:][::-1]
                angle1 = angle[no_of_scan + 1:][::-1]
                for i in range(len(rotations1)):
                    matr1 = matr1 * R.from_euler(rotations1[i], angle1[i])

                matr3 = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
                rotations3 = rotations[:no_of_scan][::-1]
                angle3 = angle[:no_of_scan][::-1]
                for i in range(len(rotations3)):
                    matr3 = matr3 * R.from_euler(rotations3[i], angle3[i])

                return (matr1.as_matrix(), matr3.as_matrix())
            return None

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
    def create_d_array(parameters: np.ndarray,
                       cell_vol: float,
                       hkl_array: np.ndarray) -> np.ndarray:
        parameters = [float(parameter) for parameter in parameters]
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
            scan_sweep: float = 0,
            start_angle: float = 0,
            epsilon: float = 1e-12,
            start_rad: Union[None, float] = None,
            end_rad: Union[None, float] = None
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

    @staticmethod
    def angles_in_sweep(
            angles_array: np.ndarray,
            start: float,
            end: float,
            sweep_type: str = 'in',
            return_bool: bool = False) -> np.ndarray:
        angles = angles_array.copy()

        if end - start >= 2 * np.pi:
            if return_bool:
                return np.ones_like(angles, dtype=bool)
            return angles

        if sweep_type == 'in':
            mask = (angles >= start) & (angles <= end)
        elif sweep_type == 'ex':
            mask = (angles < start) | (angles > end)
        else:
            raise ValueError("sweep_type must be either 'in' or 'ex'")

        if return_bool:
            return mask

        result = angles.copy()
        result[~mask] = np.nan
        return result

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

            hkl_array, original_hkl = generate_hkl_by_pg(hkl_orig_array, pg_key)

            if return_origin is False:
                return hkl_array
            elif return_origin is True:
                return hkl_array, original_hkl
        else:
            return None

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
             only_angles: bool = False,
             lorentz_minimum: Optional[float] = 0.0) -> Union[Tuple[np.ndarray, np.ndarray],
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

        angle_start = angles[no_of_scan]
        range_ = self.angle_range(scan_sweep, angle_start)
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

        angle1 = self.angles_in_sweep(angle1, range_[0], range_[1], range_[2])
        angle2 = self.angles_in_sweep(angle2, range_[0], range_[1], range_[2])
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

        if angle1.shape != (0, 1):
            diff_vec1 = apply_rotation_vecs(rotations=rotations, no_of_scan=no_of_scan, hkl_rotated=hkl_rotated1,
                                            angles=angle1, directions=directions, wavelength=wavelength, matr1=matr1,
                                            matr3=matr3)
        else:
            diff_vec1 = np.array([]).reshape(-1, 3)

        if angle2.shape != (0, 1):
            diff_vec2 = apply_rotation_vecs(rotations=rotations, no_of_scan=no_of_scan, hkl_rotated=hkl_rotated2,
                                            angles=angle2, directions=directions, wavelength=wavelength, matr1=matr1,
                                            matr3=matr3)
        else:
            diff_vec2 = np.array([]).reshape(-1, 3)

        angles_all = np.vstack((angle1, angle2))

        hkl_all = np.vstack((hkl_array1, hkl_array2)).astype(np.int32)
        hkl_array_orig_all = np.vstack((hkl_array_orig1, hkl_array_orig2))
        diff_vec_all = np.vstack((diff_vec1, diff_vec2))
        diff_vec_all = diff_vec_all / np.linalg.norm(diff_vec_all, axis=1).reshape(-1, 1)
        if lorentz_minimum and lorentz_minimum != 0.0:
            diff_vec_all, hkl_all, hkl_array_orig_all, angles_all = self._apply_lorentz_filter(
                diff_vec_all,
                hkl_all,
                hkl_array_orig_all,
                angles_all,
                rotations,
                angles,
                directions,
                no_of_scan,
                lorentz_minimum
            )
        return (diff_vec_all, hkl_all, hkl_array_orig_all, angles_all)

    def _apply_lorentz_filter(
            self,
            diff_vec_all: np.ndarray,
            hkl_all: np.ndarray,
            hkl_array_orig_all: np.ndarray,
            angles_all: np.ndarray,
            rotations: str,
            angles: Tuple[float, float, float],
            directions: Tuple[int, int, int],
            no_of_scan: int,
            lorentz_minimum: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Применяет фильтрацию по коэффициенту Лоренца."""
        primary_beam = np.array([1., 0., 0.])
        rotation_axis = self.calc_scan_vect(
            rotations=rotations,
            angles=angles,
            directions=directions,
            n_of_scan=no_of_scan
        )
        rec_vectors = np.matmul(self.orient_matx, hkl_all.reshape(-1, 3, 1)).reshape(-1, 3)
        lor_coeff = calc_lorentz(rotation_axis, diff_vec_all, rec_vectors, primary_beam)
        mask = lor_coeff > lorentz_minimum

        return (
            diff_vec_all[mask, :],
            hkl_all[mask, :],
            hkl_array_orig_all[mask, :],
            angles_all[mask, :]
        )

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

    def mapv2(self,
              reflections: np.ndarray,
              directions: Tuple[int],
              angles: Tuple[float],
              all_rotations: Tuple[str],
              yxz_rotations: Tuple[int],
              x_values: np.ndarray,
              z_values: np.ndarray,
              wavelength: float,
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        angles_ = list(angles)
        diffraction_vecs_list = []
        diffraction_angles_list = []
        hkl_list = []
        anglesx_list = []
        anglesz_list = []
        for anglez, anglex in zip(z_values, x_values):
            angles_[yxz_rotations[2]] = anglez
            angles_[yxz_rotations[1]] = anglex

            diff_vecs, hkl_array, _, angles_array = self.scan(scan_type='???', scan_sweep=360, rotations=all_rotations,
                                                              angles=angles_, directions=directions,
                                                              no_of_scan=yxz_rotations[0],
                                                              hkl_array=reflections, hkl_array_orig=reflections,
                                                              wavelength=wavelength)
            diffraction_vecs_list.append(diff_vecs)
            diffraction_angles_list.append(angles_array)
            hkl_list.append(hkl_array)
            n_data = diff_vecs.shape[0]
            anglesx_list.append(np.full(n_data, anglex).reshape(-1, 1))
            anglesz_list.append(np.full(n_data, anglez).reshape(-1, 1))

        diff_vecs_array = np.vstack(diffraction_vecs_list)
        diff_angles_array = np.vstack(diffraction_angles_list)
        hkl_array = np.vstack(hkl_list)
        anglesx = np.vstack(anglesx_list)
        anglesz = np.vstack(anglesz_list)
        data_in = (diff_vecs_array, diff_angles_array, hkl_array, anglesx, anglesz)
        return data_in

    def map_2d_v2(self,
                  reflections: np.ndarray,
                  directions: Tuple[int],
                  angles: Tuple[float],
                  all_rotations: Tuple[str],
                  yx_rotations: Tuple[int],
                  x_values: np.ndarray,
                  wavelength: float,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        angles_ = list(angles)
        diffraction_vecs_list = []
        diffraction_angles_list = []
        hkl_list = []
        anglesx_list = []
        for anglex in x_values:
            angles_[yx_rotations[1]] = anglex

            diff_vecs, hkl_array, _, angles_array = self.scan(scan_type='???', scan_sweep=360, rotations=all_rotations,
                                                              angles=angles_, directions=directions,
                                                              no_of_scan=yx_rotations[0],
                                                              hkl_array=reflections, hkl_array_orig=reflections,
                                                              wavelength=wavelength)
            diffraction_vecs_list.append(diff_vecs)
            diffraction_angles_list.append(angles_array)
            hkl_list.append(hkl_array)
            n_data = diff_vecs.shape[0]
            anglesx_list.append(np.full(n_data, anglex).reshape(-1, 1))
        diff_vecs_array = np.vstack(diffraction_vecs_list)
        diff_angles_array = np.vstack(diffraction_angles_list)
        hkl_array = np.vstack(hkl_list)
        anglesx = np.vstack(anglesx_list)
        data_in = (diff_vecs_array, diff_angles_array, hkl_array, anglesx)
        return data_in

    def map_1d_v2(self,
                  reflections: np.ndarray,
                  original_hkl: np.ndarray,
                  all_rotations: str,
                  directions: Tuple[int, int, int],
                  angles: Tuple[float, float, float],
                  scan_axis: int,
                  wavelength: float,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:

        diff_vecs, hkl_array, hkl_orig_array, angles_array = self.scan(scan_type='???', scan_sweep=360,
                                                                       rotations=all_rotations, angles=angles,
                                                                       directions=directions,
                                                                       no_of_scan=scan_axis,
                                                                       hkl_array=reflections,
                                                                       hkl_array_orig=original_hkl,
                                                                       wavelength=wavelength)
        angles_array[angles_array == 360.] = 0.
        return diff_vecs, hkl_array, hkl_orig_array, angles_array

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
