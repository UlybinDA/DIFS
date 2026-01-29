import services.service_functions as sf
from logger import logger
from services.exceptions.exceptions import CollisionError
from assets.modals_content import calc_collision_error, MODAL_TUPLE
from functools import cache
import numpy as np


class LogicCollision:
    def __init__(self, upd_on_ch: bool):
        self.rules = None
        self.active = False
        self.axes_names = None
        self.axes_rotations = None
        self.upd_on_ch = upd_on_ch

    def _update_parameters(self, exp_inst):


        self.axes_names = exp_inst.axes_names

    @cache
    def _math_context(self):
        import math
        math_context = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')}
        math_context['abs'] = abs
        return math_context


    def _calculate_limit_logic(self, scan, default_max=360.0):
        """
        Внутренний метод: принимает структуру скана (список) и считает Max Sweep.
        Используется и для одиночной проверки, и для построения карты.
        """
        scan_name = self.axes_names[int(scan[4])]
        angles_dict = dict(zip(self.axes_names, scan[3]))


        math_context = self._math_context()

        variables_dict = {
            'd_dist': scan[0],
            'det_ang_x': scan[1][0],
            'det_ang_y': scan[1][1],
            'det_ang_z': scan[1][2],
            'det_orient': scan[2],
            'scan_ax': scan_name,

            'det_disp_y': scan[6],
            'det_disp_z': scan[7]
        }
        full_context = {**math_context, **variables_dict, **angles_dict}

        current_max_sweep = default_max

        for block in self.rules:
            try:

                if block['pre_condition']:
                    if not all(sf.logic_eval(check, full_context) for check in block['pre_condition']):
                        continue
            except Exception as e:
                logger.error(f"{e}")
                continue


            for subblock in block['condition']:
                for check_str in subblock:
                    if 'sweep' not in check_str:

                        try:
                            if not sf.logic_eval(check_str, full_context):
                                return 0.0
                        except Exception as e:
                            logger.error(f"{e}")
                            pass
                    else:

                        limit = self._solve_sweep_limit(check_str, full_context)
                        if limit is not None:
                            current_max_sweep = min(current_max_sweep, limit)

        return max(0.0, current_max_sweep)



    def calculate_max_sweep(self, exp_inst, scan_index=0, default_max=360.0):
        """
        Считает sweep для конкретного существующего скана в эксперименте.
        """
        if self.upd_on_ch:
            assert exp_inst is not None
            self._update_parameters(exp_inst)

        if not self.active or self.rules is None:
            return default_max

        scan = exp_inst.scans[scan_index]

        return self._calculate_limit_logic(scan, default_max)

    def generate_sweep_map(self, exp_inst, scan, step=1.0, default_max=360.0):
        """
        Генерирует карту допустимых sweep.
        Оптимизация: если в нулевой точке разрешен полный оборот, возвращаем результат без цикла.
        Диапазон генерации: -360 ... +360.
        """
        if self.upd_on_ch:
            assert exp_inst is not None
            self._update_parameters(exp_inst)


        angles_range = np.arange(-360, 360 + step, step)


        if not self.active or self.rules is None:
            return [(angle, default_max) for angle in angles_range]

        scan_axis_idx = int(scan[4])
        temp_scan = list(scan)
        base_angles = list(scan[3])




        base_angles[scan_axis_idx] = 0.0
        temp_scan[3] = base_angles

        zero_point_limit = self._calculate_limit_logic(temp_scan, default_max)


        if zero_point_limit >= default_max:
            return [(angle, default_max) for angle in angles_range]


        result_map = []



        for angle in angles_range:
            base_angles[scan_axis_idx] = float(angle)
            temp_scan[3] = base_angles

            limit = self._calculate_limit_logic(temp_scan, default_max)


            if limit > 360.0:
                limit = 360.0

            result_map.append((angle, limit))

        return result_map

    def check(self, exp_inst=None, scans=None, type_='highest'):
        if self.upd_on_ch:
            assert exp_inst is not None
            self._update_parameters(exp_inst)

        if not self.active or self.rules is None:
            return

        scans = exp_inst.scans if scans is None else scans
        global_flag = True
        collisions = {}

        math_context = self._math_context()

        for scan_n, scan in enumerate(scans):
            scan_name = self.axes_names[int(scan[4])]
            angles_dict = dict(zip(self.axes_names, scan[3]))


            variables_dict = {
                'd_dist': scan[0],
                'det_ang_x': scan[1][0],
                'det_ang_y': scan[1][1],
                'det_ang_z': scan[1][2],
                'det_orient': scan[2],
                'scan_ax': scan_name,
                'sweep': scan[5],
                'det_disp_y': scan[6],
                'det_disp_z': scan[7]
            }
            full_context = {**math_context, **variables_dict, **angles_dict}

            for n_block, block in enumerate(self.rules):

                if (not block['pre_condition'] or
                        all(sf.logic_eval(check, full_context) for check in block['pre_condition'])):

                    block_flag = True
                    for subblock in block['condition']:
                        subblock_flag = True
                        for check in subblock:
                            element_flag = sf.logic_eval(check, full_context)
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

    def generate_boolean_mask(self, exp_inst, angles, names, initial_angles, detector):
        if self.upd_on_ch:
            assert exp_inst is not None
            self._update_parameters(exp_inst)

        if detector:
            detector_dict = {
                'd_dist': detector.dist, 'det_ang_x': detector.rot[0],
                'det_ang_y': detector.rot[1], 'det_ang_z': detector.rot[2],
                'det_orient': detector.orientation, 'det_disp_y': detector.disp_y,
                'det_disp_z': detector.disp_z
            }
        else:
            detector_dict = {}

        angles_dict = dict(zip(self.axes_names, initial_angles))
        for angles_, name in zip(angles, names):
            angles_dict[name] = angles_

        additional_operations = {'abs': abs}
        variables_dict = {**detector_dict, **angles_dict, **additional_operations}

        collisions = [i for i in self.rules if i.get('static_angle', False)]

        overall_mask = True
        for n_block, block in enumerate(collisions):
            try:
                if (not block['pre_condition'] or
                        all(sf.logic_eval(check, variables_dict) for check in block['pre_condition'])):
                    block_bool_mask = True
                    for subblock in block['condition']:
                        subblock_bool_mask = True
                        for check in subblock:
                            try:
                                check_mask = sf.logic_eval(check, variables_dict)
                                subblock_bool_mask &= check_mask
                            except:
                                subblock_bool_mask &= True
                        block_bool_mask &= subblock_bool_mask
                    overall_mask &= block_bool_mask
            except:
                overall_mask &= True
        return overall_mask

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

    def _solve_sweep_limit(self, expression, context):
        expr = expression.replace(" ", "")

        if '<=' in expr:
            parts = expr.split('<=')
        elif '<' in expr:
            parts = expr.split('<')
        else:
            return None

        lhs_str, rhs_str = parts[0], parts[1]

        if 'sweep' not in lhs_str:
            if 'sweep' not in rhs_str:
                logger.error('No sweep var in collision string')
                return None
            lhs_str, rhs_str = rhs_str, lhs_str
        if 'sweep' in rhs_str:
            logger.error('Sweep at both sides of expression')
            return None

        try:
            rhs_val = sf.safe_eval(rhs_str, context)
            context_zero_sweep = context.copy()
            context_zero_sweep['sweep'] = 0.0

            lhs_offset = sf.safe_eval(lhs_str, context_zero_sweep)

            limit = rhs_val - lhs_offset

            return float(limit)

        except Exception as e:
            logger.error(e)
            return None