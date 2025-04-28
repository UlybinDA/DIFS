import numpy as np
from pointsymmetry import PG_KEYS
from Exceptions import CollisionError, InstrumentError, NoScanDataError
from Modals_content import instrument_error_no_goniometer, calc_collision_error, MODAL_TUPLE, no_scan_data_to_save_error
from main import Sample, Ray_obstacle, DiamondAnvil
import warnings
import json
import app_gens as apg
from my_logger import mylogger
from typing import Optional, List, Tuple, Dict, Any, Union, Callable


class Experiment():
    def __init__(self):
        self.hkl_origin_in_d_range = None
        self.hkl_in_d_range = None
        self.d_range = None
        self.centring = None
        self.pg = None
        self.obstacles = list()
        self.wavelength = None
        self.scans = list()
        self.centring_previous = None
        self.pg_previous = None
        self.parameters = None
        self.axes_names = None
        self.axes_rotations = None
        self.axes_directions = None
        self.axes_real = None
        self.axes_angles = None
        self.logic_collision = None
        self.check_logic_collision = False
        self.det_geometry = None
        self.det_diameter = None
        self.det_width = None
        self.det_height = None
        self.det_complex = False
        self.det_complex_format = None
        self.det_row_col_spacing = None
        self.scan_data = None
        self.diamond_anvil = None
        self.calc_anvil_flag = False
        self.data_anvil_flag = True
        self.incident_beam_vec = np.array([1.,0.,0.])

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength

    def runs_are_set(self):
        if self.scans:
            return True
        else:
            return False

    def set_cell(self, matr=None, parameters=None, om_chi_phi=None):
        if all((matr is not None, parameters is not None)): warnings.warn(
            'orientation matrix and parameters are entered, only matrix will be used ')
        if matr is not None:
            self.cell = Sample(orient_matx=matr)
            print(matr)
            self.parameters = self.cell.parameters
        else:
            self.parameters = parameters
            a, b, c, al, bt, gm = parameters
            om, chi, phi = om_chi_phi
            self.cell = Sample(orient_matx=matr, a=a, b=b, c=c, al=al, bt=bt, gm=gm, om=om, phi=phi, chi=chi)

    def set_centring(self, centring):
        self.centring = centring

    def set_pg(self, pg):
        self.pg = pg

    def set_goniometer(self, axes_rotations=None, axes_directions=None, axes_real=None, axes_angles=None,
                       axes_names=None):
        self.clear_attributes(('axes_names'))
        self.axes_rotations = axes_rotations
        self.axes_directions = [1 if d == '+' else -1 if d == '-' else d for d in axes_directions]
        self.axes_real = axes_real
        self.axes_angles = axes_angles
        self.axes_names = axes_names

    def goniometer_is_set(self):
        if self.axes_rotations or self.axes_rotations == '':
            return True
        else:
            return False

    @mylogger(log_args=True)
    def set_detector_param(self, det_geometry, det_height=None, det_width=None, det_diameter=None, det_complex=False,
                           det_complex_format=None, det_row_col_spacing=None):
        assert det_geometry == 'rectangle' or det_geometry == 'circle', 'geometry should be rectangle or circle'
        self.det_geometry = det_geometry
        if det_geometry == 'rectangle':
            self.det_height, self.det_width, self.det_diameter = det_height, det_width, None
            if det_complex:
                self.det_complex = True
                det_complex_format = tuple(int(i) for i in det_complex_format)
                self.det_complex_format, self.det_row_col_spacing = det_complex_format, det_row_col_spacing
            else:
                self.det_complex = False
                self.det_complex_format, self.det_row_col_spacing = None, None
        if det_geometry == 'circle':
            self.det_height, self.det_width, self.det_diameter = None, None, det_diameter
            self.det_complex = False
            self.det_complex_format, self.det_row_col_spacing = None, None

    def add_obstacles(self, distance, geometry, orientation, rot, displacement_y=None, displacement_z=None, height=None,
                      width=None, diameter=None, name=''):
        if orientation == 'normal':
            displacement_y = 0
            displacement_z = 0
        if geometry == 'circle':
            height = None
            width = None
        else:
            diameter = None
        obstacle = [
            [distance, geometry, orientation, displacement_y, displacement_z, height, width, diameter, rot, name], ]

        if hasattr(self, 'obstacles'):
            self.obstacles += obstacle
        else:
            self.obstacles = []
            self.obstacles.append(obstacle)

    def delete_obstacle(self, index_of_obstacle):
        try:
            self.obstacles.pop(index_of_obstacle)
            if len(self.obstacles) == 0: self.clear_attributes('obstacles')
        except IndexError:
            warnings.warn('Index of scan is out range!')
        except AttributeError:
            warnings.warn('List of constant obstacles is empty')

    def add_scan(self, det_dist, det_angles, det_orientation, axes_angles, scan, sweep, det_disp_y=None,
                 det_disp_z=None):
        scan = (det_dist, det_angles, det_orientation, axes_angles, scan, sweep, det_disp_y, det_disp_z)

        if hasattr(self, 'scans'):
            self.scans.append(scan)
        else:
            self.scans = [scan, ]

    def delete_scan(self, index_of_scan=0, all_=False):
        if all_:
            self.scans = []
        try:
            self.scans.pop(index_of_scan)
            if len(self.scans) == 0: self.clear_attributes('scans')
        except:
            pass

    def clear_attributes(self, attributes):
        for attribute in attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)
            else:
                pass

    def create_report(self, data):
        report = ''
        for i in data:
            check_failed = f'Run {i[0]} failed the following checks :\n' + '\n'.join(
                [', '.join(j) for j in i[1]]) + '\n'
            report += check_failed
        return report

    @mylogger(level='DEBUG', log_args=True)
    def calc_experiment(self, d_range=None, hkl_section=None):
        if self.check_logic_collision and self.logic_collision is not None:
            self.check_collision_v2()
        if d_range and hkl_section is None or not None:
            pass
        if d_range and hkl_section is None:

            self.hkl_in_d_range, self.hkl_origin_in_d_range = self.cell.gen_hkl_arrays(type='d_range',
                                                                                       d_range=d_range,
                                                                                       return_origin=True,
                                                                                       pg=self.pg,
                                                                                       centring=self.centring)
            print(f'n of independent {len(np.unique(self.hkl_origin_in_d_range,axis=0))}')

            if self.diamond_anvil and self.calc_anvil_flag:
                self.hkl_in_d_range, self.hkl_origin_in_d_range = \
                    self.diamond_anvil.check_all_possible_anvil(ub_matr=self.cell.orient_matx, hkl=self.hkl_in_d_range,
                                                                data=(self.hkl_in_d_range, self.hkl_origin_in_d_range),
                                                                wavelength=self.wavelength, separate_back=False)[
                                                                                                        'all_windows']
            print(f'n of independent in bublic {len(np.unique(self.hkl_origin_in_d_range,axis=0))}')
            self.centring_previous = self.centring
            self.pg_previous = self.pg
            self.d_range = d_range

            hkl_in = self.hkl_in_d_range
            hkl_origin_in = self.hkl_origin_in_d_range

        elif hkl_section is not None:
            self.hkl_in_section = self.cell.gen_hkl_arrays(type=hkl_section)
            hkl_in = self.hkl_in_section
            hkl_origin_in = self.hkl_in_section
            pass

        all_scan_data = tuple()
        for scan_n, scan in enumerate(self.scans):
            data = self.cell.scan(scan_type='???', no_of_scan=scan[4],
                                         scan_sweep=scan[5],
                                         wavelength=self.wavelength,
                                         angles=scan[3], hkl_array=hkl_in,
                                         hkl_array_orig=hkl_origin_in,
                                         directions=self.axes_directions,
                                         rotations=self.axes_rotations)

            if self.diamond_anvil and self.calc_anvil_flag:
                data = self.diamond_anvil.filter_anvil(diff_vecs=data[0],
                                                diff_angles=data[3],
                                                rotation_axes=self.axes_rotations,
                                                directions_axes=self.axes_directions,
                                                initial_axes_angles= scan[3],
                                                scan_axis_index=scan[4],
                                                data=data,
                                                mode='transmit',
                                                incident_beam=self.incident_beam_vec
                )


            if self.det_geometry is not None:
                detector = Ray_obstacle(dist=scan[0], geometry=self.det_geometry, height=self.det_height, rot=scan[1],
                                        orientation=scan[2], width=self.det_width, diameter=self.det_diameter,
                                        complex=self.det_complex, complex_format=self.det_complex_format,
                                        row_col_spacing=self.det_row_col_spacing, disp_y=scan[6], disp_z=scan[7],
                                        )
                if self.det_complex:
                    data = detector.filter_complex_obstacle(diff_vecs=data[0], data=data, mode='transmit')
                    i = 0
                    for chip_data in data:
                        if i == 0:
                            diff_vecs, hkl, hkl_orig, angles = chip_data[0]
                        else:
                            diff_vecs = np.vstack((diff_vecs, chip_data[0][0]))
                            hkl = np.vstack((hkl, chip_data[0][1]))
                            hkl_orig = np.vstack((hkl_orig, chip_data[0][2]))
                            angles = np.vstack((angles, chip_data[0][3]))
                        i += 1
                    data = (diff_vecs, hkl, hkl_orig, angles)
                    data = detector.trash_dead_areas(data=data)

                else:
                    data = detector.filter(diff_vecs=data[0], data=data, mode='transmit')
                    data = detector.trash_dead_areas(data=data)

            if hasattr(self, 'obstacles') and self.obstacles != list():
                for obstacle in self.obstacles:
                    obstacle_ = Ray_obstacle(dist=obstacle[0], geometry=obstacle[1], orientation=obstacle[2],
                                             rot=obstacle[8], disp_y=obstacle[3], disp_z=obstacle[4],
                                             height=obstacle[5],
                                             width=obstacle[6], diameter=obstacle[7])
                    data = obstacle_.filter(diff_vecs=data[0], data=data, mode='shade')
            all_scan_data += (data,)

        self.scan_data = all_scan_data
        self.data_anvil_flag = True if self.data_anvil_flag and self.calc_anvil_flag else False

        if hasattr(self, 'known_space'):
            delattr(self, 'known_space')
        if hasattr(self, 'known_hkl'):
            delattr(self, 'known_hkl')
        if hasattr(self, 'known_hkl_orig'):
            delattr(self, 'known_hkl_orig')
        return True, None

    def load_hkls(self, hkl_str_list, pg=None, centring=None):
        data = list()
        if pg is None: pg = self.pg
        if centring is None: centring = self.centring
        d_min, d_max = (None, None)
        for hkl in hkl_str_list:
            hkl_array = sf.load_hklf4(hkl, trash_zero=True)
            d_array = sf.create_d_array(self.cell.parameters, hkl_array)
            d_min_ = min(d_array)
            d_max_ = max(d_array)
            hkl_array_orig = sf.generate_original_hkl_for_hkl_array(hkl_array, pg=pg, parameters=self.parameters,
                                                                    centring=centring)
            data += [(None, hkl_array, hkl_array_orig, None), ]
            if d_min is None or d_min > d_min_: d_min = d_min_
            if d_max is None or d_max > d_max_: d_max = d_max_
        d_range = (d_min, d_max)
        self.scan_data = data
        self.hkl_in_d_range, self.hkl_origin_in_d_range = self.cell.gen_hkl_arrays(type='d_range', d_range=d_range,
                                                                                   return_origin=True, pg=self.pg,
                                                                                   centring=self.centring)
        self.data_anvil_flag = False

    def set_diamond_anvil(self, aperture, anvil_normal):
        self.diamond_anvil = DiamondAnvil(normal=anvil_normal, aperture=aperture)
        pass

    def _generate_unique_scan_data(self, scan1, scan2):
        hkl1 = scan1[1]
        hkl2 = scan2[1]
        boolarr1, boolarr2 = sf.bool_not_intersecting_hkl(hkl1, hkl2)
        unique1 = [data[boolarr1[:, 0]] if not (data is None) else None for data in scan1]
        unique2 = [data[boolarr2[:, 0]] if not (data is None) else None for data in scan2]
        return unique1, unique2

    def separate_unique_common(self):
        assert self.scan_data is not None and len(self.scan_data) >= 2, 'Provide at least 2 scan'
        # TODO make appropriate error handling
        unique_lists = []
        for i, scan_main in enumerate(self.scan_data):
            scan_unique_list = []
            for j, scan_comparison in enumerate(self.scan_data):
                if i != j:
                    unique_, rest = self._generate_unique_scan_data(scan_main, scan_comparison)
                    scan_unique_list.append(unique_)
                else:
                    continue
            unique_lists.append(scan_unique_list)
        hkl_list = []
        for scan in self.scan_data:
            hkl_list.append(scan[1])
        intersection_indices = sf.bool_intersecting_elements(hkl_list)
        intersection_data = [data[intersection_indices[:, 0]] if not (data is None) else None for data in
                             self.scan_data[0]]
        new_scan_data = []
        for scan, unique_ in zip(self.scan_data, unique_lists):
            new_scan_data.extend((scan, *unique_))
        new_scan_data.append(intersection_data)
        self.scan_data = new_scan_data

    def set_logic_collision(self, collision_list):
        self.logic_collision = collision_list
        pass

    def _check_collision_input(self, collision_data):
        # TODO make a check for collision input
        pass

    def del_logic_collision(self):
        self.logic_collision = None

    @mylogger(log_args=True)
    def run_collision_check(self, scan=None):  # logic collision check
        '''
        Logic collision takes self.logic_collision which is list of sublists 'blocks' every block's element is also list
        'subblock' which elements - mathematical expression.
        During the check every element of subblock is evaluated. it is not necessary that all elements of each subblock
        are true. Block check stops at first True subblockflag setting blockflag True. To pass check every blockflag must get flag True.
        '''
        scans = self.scans if scan is None else scan
        conditions_not_met = list()
        global_flag = True
        for scan_n, scan in enumerate(scans):
            scan_name = self.axes_names[int(scan[4])]
            additional_operations = {'abs': abs}
            angles_dict = dict(zip(self.axes_names, scan[3]))
            variables_dict = {'d_dist': scan[0], 'det_ang_x': scan[1][0], 'det_ang_y': scan[1][1],
                              'det_ang_z': scan[1][2], 'det_orient': scan[2],
                              'scan_ax': scan_name, 'sweep': scan[5], 'det_d_y': scan[6], 'det_d_z': scan[7]}
            variables_dict = {**variables_dict, **angles_dict, **additional_operations}

            for n_block, block in enumerate(self.logic_collision):
                block_flag = True
                for subblock in block:
                    subblock_flag = True
                    for element in subblock:
                        element_flag = sf.logic_eval(element, variables_dict)
                        if not element_flag:
                            subblock_flag = False
                            break
                    if subblock_flag:
                        block_flag = True
                        break
                    block_flag = False
                if not block_flag:
                    conditions_not_met += [[scan_n, block]]
                    global_flag = False
        if not global_flag:
            report_body_add = self.create_report(conditions_not_met)
            report = MODAL_TUPLE(header=calc_collision_error.header,
                                 body=''.join((calc_collision_error.body, *report_body_add)))
            raise CollisionError(report)

    def show_completness_(self, runs='all'):
        if runs == 'all':
            data = sf.unite_runs_data(self.scan_data, runs='all')
        else:
            data = sf.unite_runs_data(self.scan_data, runs)
        completeness = sf.show_completness(data, self.hkl_origin_in_d_range)
        return completeness

    def show_redundancy_(self, runs='all'):
        if runs == 'all':
            data = sf.unite_runs_data(self.scan_data, runs='all')
        else:
            data = sf.unite_runs_data(self.scan_data, runs)
        redundancy = sf.show_redundancy_V2(data, self.hkl_origin_in_d_range)
        return redundancy

    def show_multiplicity_(self, runs='all'):
        if runs == 'all':
            data = sf.unite_runs_data(self.scan_data, runs='all')
        else:
            data = sf.unite_runs_data(self.scan_data, runs)
        multiplicity = sf.show_multiplicity_V2(data, self.hkl_origin_in_d_range)
        return multiplicity

    def generate_1d_result_plot(self, runs='all', completeness=True, redundancy=True, multiplicity=True):
        if runs == 'all':
            data = sf.unite_runs_data(self.scan_data, runs='all')
        else:
            data = sf.unite_runs_data(self.scan_data, runs)
        figs_1d = sf.make_line_plot_graph(scan=data, all_hkl_orig=self.hkl_origin_in_d_range,
                                          parameters=self.parameters, step=0.1,
                                          completeness=True, redundancy=True, multiplicity=True)
        self.figs_1d = figs_1d
        return figs_1d

    def visualize_hkl_data(self, data, visualise=True):
        sf.visualize_scans_known_hkl(data, all_hkl=self.hkl_in_d_range,
                                     all_hkl_orig=self.hkl_origin_in_d_range, visualise=visualise, trash_unknown=True,
                                     cryst_coord=True,
                                     b_matr=self.cell.b_matrix, )
        pass

    def generate_known_space_3d(self, visualise=False):
        print(self.data_anvil_flag)
        fig = sf.visualize_scans_space(self.scan_data, all_hkl=self.hkl_in_d_range,
                                       all_hkl_orig=self.hkl_origin_in_d_range, pg=self.pg, visualise=visualise,
                                       trash_unknown=False, cryst_coord=False, b_matr=self.cell.b_matrix,
                                       restore_hkl_by_pg=self.data_anvil_flag)
        self.known_space = fig
        return fig

    def generate_known_hkl_3d(self, visualise=False):
        fig = sf.visualize_scans_known_hkl(scans=self.scan_data, all_hkl=self.hkl_in_d_range,
                                           all_hkl_orig=self.hkl_origin_in_d_range, visualise=visualise,
                                           trash_unknown=False,
                                           cryst_coord=False, b_matr=self.cell.b_matrix, )
        self.known_hkl = fig
        return fig

    def generate_known_hkl_orig_3d(self, visualise=False):
        fig = sf.visualize_scans_known_hkl_orig(scans=self.scan_data, all_hkl_orig=self.hkl_origin_in_d_range,
                                                visualise=visualise, trash_unknown=False,
                                                cryst_coord=False, b_matr=self.cell.b_matrix, )
        self.known_hkl_orig = fig
        return fig

    def json_export(self, object_):
        objects = ['obstacles', 'detector', 'goniometer', 'wavelength', 'instrument', 'runs']
        assert object_ in objects, f'object of export may be: {(", ").join(objects)}'
        if object_ == 'obstacles':
            data_ = sf.generate_dicts_for_obst(exp_inst=self)
        elif object_ == 'detector':
            data_ = sf.generate_det_dict(exp_inst=self)
        elif object_ == 'wavelength':
            data_ = {'wavelength': self.wavelength}
        elif object_ == 'goniometer':
            data_ = sf.generate_dicts_for_goniometer(exp_inst=self)
        elif object_ == 'instrument':
            data_ = {
                'wavelength': {'wavelength': self.wavelength},
                'goniometer': sf.generate_dicts_for_goniometer(exp_inst=self),
                'detector': sf.generate_det_dict(exp_inst=self),
                'obstacles': sf.generate_dicts_for_obst(exp_inst=self)
            }
        elif object_ == 'runs':
            data_ = sf.generate_scans_dicts_list(exp_inst=self)
        data_json = json.dumps(data_, ensure_ascii=False, indent=4)

        return data_json

    @mylogger(log_args=True)
    def load_instrument_unit(self, data_, object_, extra=None):
        objects = ['obstacles', 'detector', 'goniometer', 'wavelength']
        assert object_ in objects, f'object of export may be: {(", ").join(objects)}'
        if object_ == 'obstacles':
            return self._load_obstacles(data_, extra)
        if object_ == 'wavelength':
            return self._load_wavelength(data_)
        if object_ == 'detector':
            return self._load_detector(data_)
        if object_ == 'goniometer':
            return self._load_goniometer(data_)

    def _load_wavelength(self, data_):
        dict_check = sf.check_dict_value(data_, 'wavelength', higher=0, classes=[float, int])
        if dict_check:
            wavelength = data_['wavelength']
            self.set_wavelength(wavelength=wavelength)
            return wavelength
        else:
            return False

    def _load_obstacles(self, data_, table_num):
        dicts_check = all([sf.check_obstacle_dict(obst) for obst in data_])
        if dicts_check:
            table_data = []
            for obst in data_:
                table = apg.generate_obst_table(n_cl=table_num, data=obst)
                table_num += 1
                rot = (obst['rotation_x'], obst['rotation_y'], obst['rotation_z'])
                obst.pop('rotation_x')
                obst.pop('rotation_y')
                obst.pop('rotation_z')
                self.add_obstacles(**obst, rot=rot)
                table_data.append(table)
        else:
            return False
        return table_data, table_num

    def _load_detector(self, data_):
        dict_check = sf.check_detector_dict(dict_=data_)
        if dict_check:
            self.set_detector_param(**data_)
            if data_['det_geometry'] == 'rectangle':
                if not data_.get('det_complex', None):
                    data = sf.generate_data_for_detector_table(width=data_['det_width'], height=data_['det_height'])
                    return data, 'Rectangle'
                else:
                    data = sf.generate_data_for_detector_table(width=data_['det_width'], height=data_['det_height'])
                    return data, 'Rectangle'
            else:
                data = sf.generate_data_for_detector_table(diameter=data_['det_diameter'])
            return data, 'Circle'
        return False

    def form_scan_data_as_hkl(self):
        if not self.scan_data:
            raise NoScanDataError(modal=no_scan_data_to_save_error)

        hkl_arrays_list = [scan[1] for scan in self.scan_data]
        hkl_array = np.vstack(hkl_arrays_list)
        lines = []
        for hkl in hkl_array:
            line = f'{hkl[0]: >4}{hkl[1]: >4}{hkl[2]: >4}{1.00: >8}{1.00: >8}\n'
            lines.append(line)

        return ''.join(lines)

    def _load_goniometer(self, data_):
        dicts_check = all((sf.check_goniometer_dict(axis) for axis in data_))
        if dicts_check:
            for dict_ in data_:
                dict_['axes_directions'] = 1 if dict_['axes_directions'] == '+' else -1
            goniometer_dict = sf.list_of_dicts_to_dict(data_, {
                'axes_rotations': str,
                'axes_directions': list,
                'axes_real': list,
                'axes_angles': list,
                'axes_names': list,
            }, )
            axes_real = goniometer_dict['axes_real']
            axes_angles = goniometer_dict['axes_angles']
            self.set_goniometer(**goniometer_dict)
            self.scans = list()
            table_data = sf.form_data_for_gon_table(axes_data=goniometer_dict)
            return table_data, axes_real, axes_angles
        else:
            return False

    @mylogger(log_args=True)
    def load_instrument(self, data_, extra=None):
        wavelength = data_.get('wavelength', None)
        goniometer = data_.get('goniometer', None)
        detector = data_.get('detector', None)
        obstacles = data_.get('obstacles', None)
        data_output = {}
        if wavelength:
            if wavelength['wavelength']:
                data_wavelength = self.load_instrument_unit(wavelength, 'wavelength')
                data_output['wavelength'] = data_wavelength
            else:
                data_output['wavelength'] = None
        else:
            data_output['wavelength'] = None
        if goniometer:
            if goniometer[0]['axes_rotations']:
                data_goniometer = self.load_instrument_unit(goniometer, 'goniometer')
                data_output['goniometer'] = data_goniometer
            else:
                data_output['goniometer'] = None
        else:
            data_output['goniometer'] = None
        if detector:
            if detector['det_geometry']:
                data_detector = self.load_instrument_unit(detector, 'detector')
                data_output['detector'] = data_detector
            else:
                data_output['detector'] = None
        else:
            data_output['detector'] = None
        if obstacles:
            if obstacles[0]['distance']:
                data_obstacles = self.load_instrument_unit(obstacles, 'obstacles', extra=extra)
                data_output['obstacles'] = data_obstacles
            else:
                data_output['obstacles'] = None
        else:
            data_output['obstacles'] = None
        return data_output

    def check_collision_v2(self, scans=None, type_='highest'):
        scans = self.scans if scans is None else scans

        global_flag = True
        collisions = {}
        for scan_n, scan in enumerate(scans):
            scan_name = self.axes_names[int(scan[4])]
            additional_operations = {'abs': abs}
            angles_dict = dict(zip(self.axes_names, scan[3]))
            variables_dict = {'d_dist': scan[0], 'det_ang_x': scan[1][0], 'det_ang_y': scan[1][1],
                              'det_ang_z': scan[1][2], 'det_orient': scan[2],
                              'scan_ax': scan_name, 'sweep': scan[5], 'det_d_y': scan[6], 'det_d_z': scan[7]}
            variables_dict = {**variables_dict, **angles_dict, **additional_operations}
            for n_block, block in enumerate(self.logic_collision):
                if (not block['pre_condition'] or
                        all(sf.logic_eval(check, variables_dict) for check in block['pre_condition'])):
                    block_flag = True
                    for subblock in block['condition']:
                        subblock_flag = True
                        for check in subblock:
                            element_flag = sf.logic_eval(check, variables_dict)
                            if not element_flag:
                                subblock_flag = False
                                break
                        if subblock_flag:
                            block_flag = True
                            break
                        block_flag = False
                    if not block_flag:
                        if scan_n not in collisions:
                            collisions[scan_n] = [block]
                        else:
                            collisions[scan_n].append(block)
                        global_flag = False

        if not global_flag:
            report_body_add = self._create_collision_report(collisions, type_)
            report = MODAL_TUPLE(header=calc_collision_error.header,
                                 body=''.join((calc_collision_error.body, *report_body_add)))
            raise CollisionError(report)

    def _create_collision_report(self, report_list, type_='highest'):
        report_str = ''
        for scan in report_list.keys():
            highest_level = min((i['level'] for i in report_list[scan]))
            report_str += f'scan_{scan}:\n'
            for report in report_list[scan]:
                if type_ == 'highest' and report['level'] == highest_level:
                    report_str += f'{report["name"]}:\n{report["problem"]}\n'
                if type_ == 'all':
                    report_str += f'{report["name"]}\n{report["problem"]}\n'
        return report_str

    def load_scans(self, data_, table_num):
        if self.axes_rotations is None:
            raise InstrumentError(instrument_error_no_goniometer)

        runs_list = []
        for run_dict in data_:
            sf.check_run_dict(exp_inst=self, dict_=run_dict)

        for run in data_:
            run_dict = sf.extract_run_data_into_list(run)
            runs_list.append(run_dict)

        if self.logic_collision and self.check_logic_collision:
            sf.check_collision(self, runs_list)

        for run_dict in runs_list:
            self.add_scan(**run_dict)

        data_ = sf.generate_data_for_runs_table(data_, table_num=table_num)
        runs_tables = []
        for run in data_:
            table = apg.gen_run_table(axes_angles=self.axes_angles, real_axes=self.axes_real,
                                      rotations=self.axes_rotations, names=self.axes_names, table_num=table_num,
                                      data=run)
            table_num += 1
            runs_tables.append(table)
        return runs_tables, table_num


import service_functions as sf

if __name__ == '__main__':
    exp2 = Experiment()
    UB = np.array([
        [-0.11849422651445085, 0.0011282219480121245, -0.01823725003496405],
        [-0.01685865323149584, -0.0817301363822078, 0.05102827245551952],
        [-0.01935107006442972, 0.06386615672860567, 0.06568363644621458],
    ])
    parameters = [10, 10, 10, 90.000, 90, 90]
    # exp2.set_cell(matr=UB)
    exp2.set_cell(parameters=parameters,om_chi_phi=(0,0,0))
    exp2.set_pg('-1')
    exp2.set_centring('P')
    exp2.set_wavelength(0.710730)
    goniometer_system = 'z'
    angles = [
        [0],
        # [0, 0, 20, -90],
        # [0, 0, 40, -90],
        # [0, 0, 60, -90],
        # [0, 0, 80, -90],
    ]
    rotation_dirs = (1,)
    aperture = 40
    anvil_normal = np.array([1.,0.,0.])
    exp2.set_goniometer(goniometer_system, axes_directions=rotation_dirs, axes_real=['true'], axes_angles=[0],
                        axes_names=['a', 'b', 'omega'])
    for angle in angles:
        exp2.add_scan(det_dist=95, det_angles=[0, 0, 25], det_orientation='normal', axes_angles=angle, scan=0,
                      sweep=5, )

    # exp2.set_diamond_anvil(aperture=aperture,anvil_normal=anvil_normal)
    # exp2.calc_anvil_flag=True

    exp2.calc_experiment(d_range=(5, 11))
    data = exp2.scan_data

    # hkl_text = exp2.form_scan_data_as_hkl()
    #
    # with open('Piva_2.36GPa_in_P1.hkl', 'r') as f:
    #     a = f.read()
    # exp2.load_hkls([a],pg=-1,centring='P')
    # exp2.separate_unique_common()
    # exp2.generate_known_space_3d(visualise=True)
    # exp1.set_logic_collision(collision)
    # exp1.check_collision_v2()

    # with open('image_.hkl','r') as f:
    #     str = f.read()
    # exp2.load_hkls([str],)
    # fig = exp2.generate_known_space_3d()
    # fig.show()
