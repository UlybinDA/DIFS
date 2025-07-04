import numpy as np
import pandas as pd
import plotly.graph_objs as go
import random
import services.rgb_colors as rgb_colors
from sample.sample import Sample
from logger.my_logger import mylogger
from symmetry.pointsymmetry import PG_KEYS, multiply_hkl_by_pg, generate_hkl_by_pg, generate_orig_hkl_array
import plotly.express as px
import warnings
from scipy.spatial.transform import Rotation as R
import base64
from experiment.experiment import Experiment
import json
from services.exceptions.exceptions import RunsDictError, HKLFormatError, WrongHKLShape, CalcCumulativeCompletenessError
from assets.modals_content import *
from services.encode_hkl import encode_hkl
from typing import Tuple, List, Dict, Union, Optional, Any
import re
from obstacles.obstacle import Ray_obstacle
from obstacles.linked_obstacle import LinkedObstacle


def load_hklf4(hkl_str: str, trash_zero: bool = True) -> np.ndarray:
    try:
        lines = hkl_str.split('\n')
        hkl_data = []
        for line in lines:
            parts = line.split()
            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                hkl_data.append([h, k, l])
            except (ValueError, IndexError):
                continue

        data = np.array(hkl_data)

        if trash_zero:
            zero_mask = (data[:, 0] != 0) | (data[:, 1] != 0) | (data[:, 2] != 0)
            data = data[zero_mask]

        return data
    except:
        raise HKLFormatError(hkl_format_error)


def load_np_hklf(name: str) -> np.ndarray:
    return np.load(name)


def gen_hkl_arrays(hmax: int, kmax: int, lmax: int, pg: Optional[str] = None, centring: str = 'P') -> Tuple[
    np.ndarray, np.ndarray]:
    pg_key = PG_KEYS[pg]
    original_hkl = generate_orig_hkl_array(h=hmax, k=kmax, l=lmax, pg=pg_key, centring=centring)
    hkl_array, original_hkl = generate_hkl_by_pg(hkl_orig_array=original_hkl, pg_key=pg)
    return hkl_array, original_hkl


def cut_by_d(hkl_array: np.ndarray, d_range: List[float], parameters: List[float], bool_out: bool = False) -> Union[
    np.ndarray, np.ndarray]:
    d_array = create_d_array(parameters=parameters, hkl_array=hkl_array)
    bool_array = ((d_array >= d_range[0]) & (d_array <= d_range[1])).reshape(-1, 1)
    if bool_out:
        return bool_array
    return hkl_array[bool_array[:, 0]]


def generate_original_hkl_for_hkl_array(hkl_array: np.ndarray, pg: str, parameters: List[float],
                                        centring: str = 'P') -> np.ndarray:
    d_array = create_d_array(parameters=parameters, hkl_array=hkl_array)
    d_range = (min(d_array), max(d_array))
    sample = Sample(a=parameters[0], b=parameters[1], c=parameters[2], al=parameters[3], bt=parameters[4],
                    gm=parameters[5])
    hkl_array_, hkl_array_orig = sample.gen_hkl_arrays(type='d_range', d_range=d_range, return_origin=True, pg=pg,
                                                       centring=centring)
    generated_hkl_encoded_array = encode_hkl(hkl_array_)
    hkl_encoded_array = encode_hkl(hkl_array)
    sort_idx = np.argsort(generated_hkl_encoded_array)
    sorted_hkl = generated_hkl_encoded_array[sort_idx]
    sorted_hkl_o = hkl_array_orig[sort_idx.reshape(-1, 1)[:, 0]]
    indices = np.searchsorted(sorted_hkl, hkl_encoded_array).reshape(-1, 1)
    hkl_orig = sorted_hkl_o[indices[:, 0]]
    return hkl_orig


def save_scan_data(data_to_save: Tuple[np.ndarray, ...], name: str) -> None:
    data = np.hstack((data_to_save[0], data_to_save[1], data_to_save[2], data_to_save[3]))
    np.save(f'{name}', data)


def indices_of_common_rows(array1: np.ndarray, array2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    array1_encoded = encode_hkl(array1)
    array2_encoded = encode_hkl(array2)
    array1_indices, array2_indices = indices_of_common_elements(array1_encoded, array2_encoded)
    return array1_indices, array2_indices


def make_single_column_str_arr(array: np.ndarray) -> np.ndarray:
    scsa = np.char.add(np.char.add(np.char.add(np.char.add(array[:, 0].astype(int).astype(str), ' '),
                                               array[:, 1].astype(int).astype(str)), ' '),
                       array[:, 2].astype(int).astype(str))
    return scsa


def bool_intersecting_elements(arrays):
    hkl_arrays_encoded = [encode_hkl(hkl) for hkl in arrays]
    st_hkl_encoded = hkl_arrays_encoded[0]
    common = set(st_hkl_encoded.copy())
    for hkl in hkl_arrays_encoded[1:]:
        common.intersection_update(hkl)
        if not common:
            break
    bool_array = np.isin(st_hkl_encoded, np.array(tuple(common)))
    return bool_array.reshape(-1, 1)


def bool_not_intersecting_hkl(hkl_array1, hkl_array2):
    if not hkl_array1.shape[1] == 3 or not hkl_array1.shape[1] == 3:
        raise WrongHKLShape(modal=wrong_hkl_array_shape)
    hkl_encoded_array1 = encode_hkl(hkl_array1)
    hkl_encoded_array2 = encode_hkl(hkl_array2)
    boolarr1 = np.ones(len(hkl_encoded_array1)).astype(dtype=bool)
    boolarr2 = np.ones(len(hkl_encoded_array2)).astype(dtype=bool)
    intersection1, intersection2 = indices_of_common_elements(hkl_encoded_array1, hkl_encoded_array2)
    boolarr1[intersection1] = False
    boolarr2[intersection2] = False
    return boolarr1.reshape(-1, 1), boolarr2.reshape(-1, 1)


def indices_of_common_elements(array1: np.ndarray, array2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    intersection = np.intersect1d(array1, array2)
    array1_indices = np.where(np.isin(array1, intersection))[0]
    array2_indices = np.where(np.isin(array2, intersection))[0]
    return array1_indices, array2_indices


def generate_hkl_data_for_visual(data: Tuple[np.ndarray, ...], all_hkl: np.ndarray, all_hkl_orig: np.ndarray, pg: str,
                                 color: str = 'blue', trash_unknown: bool = False, cryst_coord: bool = True,
                                 b_matr: Optional[np.ndarray] = None) -> Dict[
    str, Union[np.ndarray, Tuple[str, ...], int]]:
    if trash_unknown == False:
        data = np.hstack(data)
        known_rec_space, hkl_orig_array = multiply_hkl_by_pg(data, pg)
        known_rec_space_encoded_array = encode_hkl(known_rec_space)
        all_hkl_encoded_array = encode_hkl(all_hkl)
        mask_for_known_rec_space = np.ones((all_hkl.shape[0], 1), dtype=bool)
        *rest, indices_for_rec_space = np.intersect1d(known_rec_space_encoded_array, all_hkl_encoded_array,
                                                      return_indices=True)

        mask_for_known_rec_space[indices_for_rec_space] = False

        not_incl_rec_space = all_hkl[mask_for_known_rec_space[:, 0]]
        not_incl_hkl_orig = all_hkl_orig[mask_for_known_rec_space[:, 0]]
        color_false_tuple = tuple(('#ff0000',) * not_incl_rec_space.shape[0])

        known_rec_space, indices = np.unique(known_rec_space, return_index=True, axis=0)
        known_hkl_orig = hkl_orig_array[indices]
        color_true_tuple = tuple((color,) * known_rec_space.shape[0])

        hkl_array = np.vstack((known_rec_space, not_incl_rec_space))
        hkl_orig_array = np.vstack((known_hkl_orig, not_incl_hkl_orig))
        hkl_orig_array_str = make_single_column_str_arr(hkl_orig_array)
        color_tuple = color_true_tuple + color_false_tuple
        if cryst_coord:
            data_out = {
                'h': hkl_array[:, 0].reshape(-1),
                'k': hkl_array[:, 1].reshape(-1),
                'l': hkl_array[:, 2].reshape(-1),
                'hkl_orig': hkl_orig_array_str.reshape(-1),
                'h_o': hkl_orig_array[:, 0].reshape(-1),
                'k_o': hkl_orig_array[:, 1].reshape(-1),
                'l_o': hkl_orig_array[:, 2].reshape(-1),
                'color': color_tuple,
                'i_first_unknown': color_tuple.index('#ff0000')
            }
        else:
            xyz_array = np.matmul(b_matr, hkl_array.reshape(-1, 3, 1)).reshape(-1, 3)
            data_out = {
                'h': hkl_array[:, 0].reshape(-1),
                'k': hkl_array[:, 1].reshape(-1),
                'l': hkl_array[:, 2].reshape(-1),
                'x': xyz_array[:, 0].reshape(-1),
                'y': xyz_array[:, 1].reshape(-1),
                'z': xyz_array[:, 2].reshape(-1),
                'h_o': hkl_orig_array[:, 0].reshape(-1),
                'k_o': hkl_orig_array[:, 1].reshape(-1),
                'l_o': hkl_orig_array[:, 2].reshape(-1),
                'color': color_tuple,
                'i_first_unknown': color_tuple.index('#ff0000')
            }
            pass


    elif trash_unknown == True:
        data = np.hstack(data)
        known_rec_space, hkl_orig_array = multiply_hkl_by_pg(data, pg)

        known_rec_space, indices = np.unique(known_rec_space, return_index=True, axis=0)
        known_hkl_orig = hkl_orig_array[indices]
        color_true_tuple = tuple((color,) * known_rec_space.shape[0])
        if cryst_coord:
            data_out = {
                'h': known_rec_space[:, 0].reshape(-1),
                'k': known_rec_space[:, 1].reshape(-1),
                'l': known_rec_space[:, 2].reshape(-1),
                'h_o': known_hkl_orig[:, 0].reshape(-1),
                'k_o': known_hkl_orig[:, 1].reshape(-1),
                'l_o': known_hkl_orig[:, 2].reshape(-1),
                'color': color_true_tuple
            }
        else:
            xyz_array = np.matmul(b_matr, known_rec_space.reshape(-1, 3, 1)).reshape(-1, 3)
            data_out = {
                'h': known_rec_space[:, 0].reshape(-1),
                'k': known_rec_space[:, 1].reshape(-1),
                'l': known_rec_space[:, 2].reshape(-1),
                'x': xyz_array[:, 0].reshape(-1),
                'y': xyz_array[:, 1].reshape(-1),
                'z': xyz_array[:, 2].reshape(-1),
                'h_o': known_hkl_orig[:, 0].reshape(-1),
                'k_o': known_hkl_orig[:, 1].reshape(-1),
                'l_o': known_hkl_orig[:, 2].reshape(-1),
                'color': color_true_tuple
            }
    return data_out


def generate_hkl_orig_data_for_visual(data: Tuple[np.ndarray, ...], all_orig_hkl: np.ndarray, color: str,
                                      trash_unknown: bool, cryst_coord: bool = True,
                                      b_matr: Optional[np.ndarray] = None) -> Dict[
    str, Union[np.ndarray, Tuple[str, ...], int]]:
    if trash_unknown is False:
        known_hkl_orig = np.unique(data[1], axis=0)
        known_hkl_orig_encoded_array = (encode_hkl(known_hkl_orig))
        all_orig_hkl = np.unique(all_orig_hkl, axis=0)
        all_orig_hkl_encoded_array = encode_hkl(all_orig_hkl)
        mask = np.ones((all_orig_hkl.shape[0], 1), dtype=bool)
        *rest, indices = np.intersect1d(known_hkl_orig_encoded_array, all_orig_hkl_encoded_array, return_indices=True)
        mask[indices] = False

        not_included_orig_hkl = all_orig_hkl[mask[:, 0]]
        color_false_tuple = tuple(('#ff0000',) * not_included_orig_hkl.shape[0])

        hkl_array = np.vstack((known_hkl_orig, not_included_orig_hkl))
        color_true_tuple = tuple((f'{color}',) * known_hkl_orig.shape[0])
        color_tuple = color_true_tuple + color_false_tuple
        if cryst_coord:
            data = {
                'h': hkl_array[:, 0].reshape(-1),
                'k': hkl_array[:, 1].reshape(-1),
                'l': hkl_array[:, 2].reshape(-1),
                'h_o': hkl_array[:, 0].reshape(-1),
                'k_o': hkl_array[:, 1].reshape(-1),
                'l_o': hkl_array[:, 2].reshape(-1),
                'color': color_tuple,
                'i_first_unknown': color_tuple.index('#ff0000')
            }
        else:
            xyz_array = np.matmul(b_matr, hkl_array.reshape(-1, 3, 1)).reshape(-1, 3)
            data = {
                'x': xyz_array[:, 0].reshape(-1),
                'y': xyz_array[:, 1].reshape(-1),
                'z': xyz_array[:, 2].reshape(-1),
                'h': hkl_array[:, 0].reshape(-1),
                'k': hkl_array[:, 1].reshape(-1),
                'l': hkl_array[:, 2].reshape(-1),
                'h_o': hkl_array[:, 0].reshape(-1),
                'k_o': hkl_array[:, 1].reshape(-1),
                'l_o': hkl_array[:, 2].reshape(-1),
                'color': color_tuple,
                'i_first_unknown': color_tuple.index('#ff0000')
            }
    elif trash_unknown is True:
        hkl_array = np.unique(data[1], axis=0)

        color_tuple = tuple((f'{color}',) * hkl_array.shape[0])
        if cryst_coord:
            data = {
                'h': hkl_array[:, 0].reshape(-1),
                'k': hkl_array[:, 1].reshape(-1),
                'l': hkl_array[:, 2].reshape(-1),
                'h_o': hkl_array[:, 0].reshape(-1),
                'k_o': hkl_array[:, 1].reshape(-1),
                'l_o': hkl_array[:, 2].reshape(-1),
                'color': color_tuple
            }
        else:
            xyz_array = np.matmul(b_matr, hkl_array.reshape(-1, 3, 1)).reshape(-1, 3)
            data = {
                'x': xyz_array[:, 0].reshape(-1),
                'y': xyz_array[:, 1].reshape(-1),
                'z': xyz_array[:, 2].reshape(-1),
                'h': hkl_array[:, 0].reshape(-1),
                'k': hkl_array[:, 1].reshape(-1),
                'l': hkl_array[:, 2].reshape(-1),
                'h_o': hkl_array[:, 0].reshape(-1),
                'k_o': hkl_array[:, 1].reshape(-1),
                'l_o': hkl_array[:, 2].reshape(-1),
                'color': color_tuple
            }

        pass
    return data


def visualize_scans_space(scans: List[Tuple[np.ndarray, ...]], pg: str, all_hkl: Optional[np.ndarray] = None,
                          all_hkl_orig: Optional[np.ndarray] = None, colors: Optional[List[str]] = None,
                          trash_unknown: bool = True,
                          origin_hkl: bool = False, visualise: bool = True, cryst_coord: bool = False,
                          b_matr: Optional[np.ndarray] = None,
                          restore_hkl_by_pg=False
                          ) -> Optional[go.Figure]:
    if b_matr is None and not cryst_coord:
        warnings.warn('visualizations in direct space but no B matrix provided, switching to crystal coordinates')
        cryst_coord = True
        pass
    if not trash_unknown:
        pass
    scans_data = list()
    if colors is None:
        colors = rgb_colors.colors[0:len(scans[0])]
        pass
    n = -1
    if restore_hkl_by_pg:
        all_hkl, all_hkl_orig = generate_hkl_by_pg(hkl_orig_array=all_hkl_orig, pg_key=PG_KEYS[pg])

    if not origin_hkl:
        for scan in zip(*scans):
            n += 1
            scan_data = generate_hkl_data_for_visual(scan, all_hkl, all_hkl_orig, pg, color=colors[n],
                                                     trash_unknown=True, cryst_coord=cryst_coord,
                                                     b_matr=b_matr)
            scans_data += [scan_data, ]

        hkl_array = np.array([[], [], []])
        for data in scans_data:
            hkl_array = np.hstack((hkl_array, np.vstack((data['h'], data['k'], data['l']))))
        hkl_array = hkl_array.transpose()
        hkl_array_encoded = np.unique(encode_hkl(hkl_array), axis=0)
        hkl_array_all_encoded = encode_hkl(all_hkl)
        rest, intersect_indices = indices_of_common_elements(hkl_array_encoded, hkl_array_all_encoded)
        bool_array = np.ones(all_hkl.shape[0], dtype=bool)
        bool_array[intersect_indices] = False
        bool_array = bool_array.reshape(-1, 1)
        hkl_array_unknown = all_hkl[bool_array[:, 0]]
        hkl_o_array_unknown = all_hkl_orig[bool_array[:, 0]]
        xyz_array_unknown = np.matmul(b_matr, hkl_array_unknown.reshape(-1, 3, 1)).reshape(-1, 3)
        data_unknown = np.hstack((xyz_array_unknown, hkl_array_unknown, hkl_o_array_unknown))


    elif origin_hkl:
        for scan in zip(*scans):
            n += 1
            scan_data = generate_hkl_orig_data_for_visual(scan, all_orig_hkl=all_hkl_orig, color=colors[n],
                                                          trash_unknown=True, cryst_coord=cryst_coord,
                                                          b_matr=b_matr)
            scans_data += [scan_data, ]
        hkl_array_orig = np.array([[], [], []])
        for data in scans_data:
            hkl_array_orig = np.hstack((hkl_array_orig, np.vstack((data['h'], data['k'], data['l']))))
        hkl_orig = np.unique(all_hkl_orig, axis=0)
        hkl_array_orig = hkl_array_orig.transpose()
        hkl_array_orig = np.unique(hkl_array_orig, axis=0)
        hkl_array_encoded = encode_hkl(hkl_array_orig)

        hkl_array_all_o_encoded = encode_hkl(hkl_orig)
        rest, intersect_indices = indices_of_common_elements(hkl_array_encoded, hkl_array_all_o_encoded)
        bool_array = np.ones(hkl_orig.shape[0], dtype=bool)
        bool_array[intersect_indices] = False
        bool_array = bool_array.reshape(-1, 1)
        hkl_array_unknown = hkl_orig[bool_array[:, 0]]
        xyz_array_unknown = np.matmul(b_matr, hkl_array_unknown.reshape(-1, 3, 1)).reshape(-1, 3)
        data_unknown = np.hstack((xyz_array_unknown, hkl_array_unknown, hkl_array_unknown))

    if cryst_coord:
        max_index = np.amax(np.abs(all_hkl_orig))
    else:
        max_indexh = np.amax(np.abs(all_hkl_orig[:, 0]))
        max_indexk = np.amax(np.abs(all_hkl_orig[:, 1]))
        max_indexl = np.amax(np.abs(all_hkl_orig[:, 2]))
        max_indexx = np.matmul(b_matr, np.array([[max_indexh], [0], [0]])).reshape(-1)
        max_indexy = np.matmul(b_matr, np.array([[0], [max_indexk], [0]])).reshape(-1)
        max_indexz = np.matmul(b_matr, np.array([[0], [0], [max_indexl]])).reshape(-1)

    plots = list()
    num = 0
    update_menus_dict = dict()
    button_list = list()

    for scan_data in scans_data:
        num += 1
        if cryst_coord:
            plots += [go.Scatter3d(x=scan_data['h'], y=scan_data['k'], z=scan_data['l'], name=f'run {num}',
                                   hovertext=scan_data['hkl_orig'],
                                   marker={'color': scan_data['color'], 'size': 3}, mode='markers')]
        else:
            plots += [go.Scatter3d(x=scan_data['x'], y=scan_data['y'], z=scan_data['z'], name=f'run {num}',
                                   customdata=np.stack([scan_data['h'], scan_data['k'], scan_data['l'],
                                                        scan_data['h_o'], scan_data['k_o'], scan_data['l_o']], axis=-1),
                                   hovertemplate='<b>hkl</b>: %{customdata[0]}<b> %{customdata[1]}<b> %{customdata[2]}' +
                                                 '<br>hkl orig</b>: %{customdata[3]}<b> %{customdata[4]}<b> %{customdata[5]}',
                                   marker={'color': scan_data['color'], 'size': 3}, mode='markers'
                                   ), ]

        dictionary = dict()
        args_marker_list = list()
        for i in range(len(scans_data)):
            if i == num - 1:
                scan_dict = 3
            else:
                scan_dict = 1

            args_marker_list += [scan_dict]

        dictionary.update(dict(args=[{'marker.size': args_marker_list}]))
        dictionary.update(dict(label=f'run {num}'))
        dictionary.update(dict(method='update'))
        button_list += [dictionary]

    plots += [go.Scatter3d(name='unknown_points', x=data_unknown[:, 0].reshape(-1), y=data_unknown[:, 1].reshape(-1),
                           z=data_unknown[:, 2].reshape(-1),
                           customdata=data_unknown[:, 3:],
                           hovertemplate='<b>hkl</b>: %{customdata[0]}<b> %{customdata[1]}<b> %{customdata[2]}' +
                                         '<br>hkl orig</b>: %{customdata[3]}<b> %{customdata[4]}<b> %{customdata[5]}',
                           marker={"color": ('#ff0000',) * data_unknown.shape[0], 'size': 3}, mode='markers'), ]
    update_menus_dict.update(dict(buttons=button_list))
    update_menus_dict.update(dict(direction='down'))
    update_menus_dict.update(dict(showactive=True))
    update_menus_dict.update(dict(x=0.1))
    update_menus_dict.update(dict(y=0.1))
    update_menus_dict.update(dict(xanchor='left'))
    update_menus_dict.update(dict(yanchor='top'))
    layout = go.Layout(dict(updatemenus=[update_menus_dict], dragmode='turntable'))
    fig = go.Figure(data=plots, layout=layout)
    fig.add_scatter3d(name='selected_points',
                      customdata=np.array([]).reshape(0, 6),
                      x=np.array([]), y=np.array([]), z=np.array([]),
                      marker={'size': 3}, mode='markers')
    mult = 1.3
    if cryst_coord:
        fig.add_scatter3d(x=[max_index * mult, -max_index * mult], y=[0, 0], z=[0, 0], mode='lines+markers+text',
                          text=['h', '-h'],
                          marker={'color': ['red', 'red'], 'size': [0, 0]}, line={'color': 'red'})
        fig.add_scatter3d(x=[0, 0], y=[max_index * mult, -max_index * mult], z=[0, 0], mode='lines+markers+text',
                          text=['k', '-k'],
                          marker={'color': ['green', 'green'], 'size': [0, 0]}, line={'color': 'green'})
        fig.add_scatter3d(x=[0, 0], y=[0, 0], z=[max_index * mult, -max_index * mult], mode='lines+markers+text',
                          text=['l', '-l'],
                          marker={'color': ['blue', 'blue'], 'size': [0, 0]}, line={'color': 'blue'})
    else:
        fig.add_scatter3d(x=[max_indexx[0] * mult, -max_indexx[0] * mult], y=[max_indexx[1], -max_indexx[1]],
                          z=[max_indexx[2], -max_indexx[2]],
                          mode='lines+markers+text',
                          text=['h', '-h'],
                          name='h axis',
                          marker={
                              'color': ['red', 'red'], 'size': (0, 0)}, line={'color': 'red'})
        fig.add_scatter3d(x=[max_indexy[0] * mult, -max_indexy[0] * mult],
                          y=[max_indexy[1] * mult, -max_indexy[1] * mult],
                          z=[max_indexy[2] * mult, -max_indexy[2] * mult],
                          mode='lines+markers+text',
                          text=['k', '-k'],
                          name='k axis',
                          marker={
                              'color': ['green', 'green'], 'size': (0, 0)}, line={'color': 'green'})
        fig.add_scatter3d(x=[max_indexz[0] * mult, -max_indexz[0] * mult],
                          y=[max_indexz[1] * mult, -max_indexz[1] * mult],
                          z=[max_indexz[2] * mult, -max_indexz[2] * mult],
                          mode='lines+markers+text',
                          text=['l', '-l'],
                          name='l axis',
                          marker={
                              'color': ['blue', 'blue'], 'size': (0, 0)}, line={'color': 'blue'})

    layout_addition = [3, 3, (0, 0), (0, 0), (0, 0)]
    for button in fig.layout['updatemenus'][0]['buttons']:
        button['args'][0]['marker.size'] += layout_addition

    fig.layout.scene.camera.projection.type = "orthographic"
    max_index = np.max([max_indexx, max_indexy, max_indexz])
    fig.update_layout(scene={
        'xaxis': {'range': [-max_index, max_index]},
        'yaxis': {'range': [-max_index, max_index]},
        'zaxis': {'range': [-max_index, max_index]},
    })

    fig.update_layout(width=1000, height=1000, scene={'aspectmode': 'cube'})

    if visualise is True:
        fig.show()
    else:
        return fig


def visualize_scans_known_hkl(scans: List[Tuple[np.ndarray, ...]], all_hkl: np.ndarray, all_hkl_orig: np.ndarray,
                              trash_unknown: bool = True, visualise: bool = True, cryst_coord: bool = False,
                              b_matr: Optional[np.ndarray] = None, ) -> Optional[go.Figure]:
    if visualise is True:
        visualize_scans_space(scans=scans, all_hkl=all_hkl, all_hkl_orig=all_hkl_orig, pg='1', colors=None,
                              trash_unknown=trash_unknown, visualise=visualise, cryst_coord=cryst_coord,
                              b_matr=b_matr, )
    else:
        fig = visualize_scans_space(scans=scans, all_hkl=all_hkl, all_hkl_orig=all_hkl_orig, pg='1', colors=None,
                                    trash_unknown=trash_unknown, visualise=visualise, cryst_coord=cryst_coord,
                                    b_matr=b_matr, )
        return fig


def visualize_scans_known_hkl_orig(scans: List[Tuple[np.ndarray, ...]], all_hkl_orig: np.ndarray,
                                   colors: Optional[List[str]] = None, trash_unknown: bool = True,
                                   visualise: bool = True, cryst_coord: bool = False,
                                   b_matr: Optional[np.ndarray] = None) -> Optional[go.Figure]:
    if visualise is True:
        visualize_scans_space(scans=scans, all_hkl=None, all_hkl_orig=all_hkl_orig, colors=colors,
                              trash_unknown=trash_unknown, pg=None, origin_hkl=True, visualise=visualise,
                              cryst_coord=cryst_coord, b_matr=b_matr, )
    else:
        fig = visualize_scans_space(scans=scans, all_hkl=None, all_hkl_orig=all_hkl_orig, colors=colors,
                                    trash_unknown=trash_unknown, pg=None, origin_hkl=True, visualise=visualise,
                                    cryst_coord=cryst_coord, b_matr=b_matr, )
        return fig


def subtract_second_run_from_first(run1: Tuple[np.ndarray, ...], run2: Tuple[np.ndarray, ...]) -> Tuple[
    Tuple[None, np.ndarray, np.ndarray, None],]:
    hkl1 = run1[1]
    hkl2 = run2[1]
    hkl1_encoded = encode_hkl(hkl1)
    hkl2_encoded = encode_hkl(hkl2)
    mask = np.invert(np.isin(hkl1_encoded, hkl2_encoded)).reshape(-1, 1)
    output_hkl = hkl1[mask[:, 0]]
    output_hkl_orig = run1[2][mask[:, 0]]
    return ((None, output_hkl, output_hkl_orig, None),)


def subtract_equivalents_second_run_from_first(run1: Tuple[np.ndarray, ...], run2: Tuple[np.ndarray, ...]) -> Tuple[
    Tuple[None, np.ndarray, np.ndarray, None],]:
    hkl1 = run1[2]
    hkl2 = run2[2]
    hkl1_encoded = encode_hkl(hkl1)
    hkl2_encoded = encode_hkl(hkl2)
    mask = np.invert(np.isin(hkl1_encoded, hkl2_encoded)).reshape(-1, 1)
    output_hkl = run1[1][mask[:, 0]]
    output_hkl_orig = hkl1[mask[:, 0]]
    return ((None, output_hkl, output_hkl_orig, None),)


def multiply_runs_hkl_by_pg(run1: Tuple[np.ndarray, ...], pg: str) -> Tuple[Tuple[None, np.ndarray, np.ndarray, None],]:
    hkl, hkl_orig = multiply_hkl_by_pg(run1[1:3], pg)
    output_hkl, hkl_indices = np.unique(hkl, axis=0, return_index=True)
    output_hkl_orig = hkl_orig[hkl_indices]
    return ((None, output_hkl, output_hkl_orig, None),)


def reduce_to_origin(run: Tuple[np.ndarray, ...]) -> Tuple[Tuple[None, np.ndarray, np.ndarray, None],]:
    hkl, hkl_orig = run[1:3]
    output_hkl_orig, hkl_indices = np.unique(hkl_orig, axis=0, return_index=True)
    output_hkl = hkl[hkl_indices]
    return ((None, output_hkl, output_hkl_orig, None),)


def unite_runs_data(all_data: List[Tuple[np.ndarray, ...]], runs: Union[str, List[int]]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if runs == 'all':
        data = all_data[:]
    else:
        data = [all_data[i] for i in runs]

    angles_list, hkl_list, hkl_original_list, diff_vecs_list = [], [], [], []

    for run in data:
        angles_list.append(run[0])
        hkl_list.append(run[1])
        hkl_original_list.append(run[2])
        diff_vecs_list.append(run[3])

    angles = np.vstack(angles_list) if angles_list else None
    hkl = np.vstack(hkl_list) if hkl_list else None
    hkl_original = np.vstack(hkl_original_list) if hkl_original_list else None
    diff_vecs = np.vstack(diff_vecs_list) if diff_vecs_list else None

    return (angles, hkl, hkl_original, diff_vecs)


def separate_diff_vectors_by_vector_direction(diff_vecs: np.ndarray, data: Tuple[np.ndarray, ...], normals: np.ndarray,
                                              ki: np.ndarray) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    mask1 = (diff_vecs[:, 0] * normals[:, 0] + diff_vecs[:, 1] * normals[:, 1] + diff_vecs[:, 2] * normals[:, 2]) > 0
    mask2 = (normals[:, 0] * ki[0] + normals[:, 1] * ki[1] + normals[:, 2] * ki[2]) > 0
    mask = (mask1 == mask2).reshape(-1, 1)

    data_front = tuple()
    data_rear = tuple()
    for array in data:
        array_shape = array.shape[1]
        data_front += (array[mask[:, 0] == True].reshape(-1, array_shape),)
        data_rear += (array[mask[:, 0] == False].reshape(-1, array_shape),)
    return (data_front, data_rear)


def show_completness(hkl_origin: np.ndarray, all_hkl_orig: np.ndarray) -> float:
    n_indepen_refs = np.unique(hkl_origin, axis=0).shape[0]
    n_all_indepen_refs = np.unique(all_hkl_orig, axis=0).shape[0]
    completeness = n_indepen_refs / n_all_indepen_refs * 100
    return completeness


def show_redundancy_V2(hkl: np.ndarray, all_hkl_orig: np.ndarray) -> float:
    n_hkl = hkl.shape[0]
    n_all_hkl_orig = all_hkl_orig.shape[0]
    redundancy = n_hkl / n_all_hkl_orig
    return redundancy


def show_multiplicity_V2(hkl_origin: np.ndarray, all_hkl_orig: np.ndarray) -> float:
    n_hkl_origin = hkl_origin.shape[0]
    n_all_hkl_orig_unique = np.unique(all_hkl_orig, axis=0).shape[0]
    multiplicity = np.sum(n_hkl_origin) / n_all_hkl_orig_unique
    return multiplicity


def calc_cell_volume(parameters: List[float]) -> float:
    cell_vol = parameters[0] * parameters[1] * parameters[2] * (1 - np.cos(np.deg2rad(parameters[3])) ** 2 - np.cos(
        np.deg2rad(parameters[4])) ** 2 - np.cos(np.deg2rad(parameters[5])) ** 2 + 2 * np.cos(
        np.deg2rad(parameters[3])) * np.cos(np.deg2rad(parameters[3])) * np.cos(np.deg2rad(parameters[3]))) ** 0.5
    return cell_vol


def generate_completeness_plot(hkl_orig: np.ndarray, all_hkl_orig: np.ndarray, parameters: List[float],
                               step: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    hkl_orig_unique = np.unique(hkl_orig, axis=0)
    hkl_orig_unique_d_array = create_d_array(parameters, hkl_orig_unique)
    all_hkl_orig_unique = np.unique(all_hkl_orig, axis=0)
    all_hkl_orig_unique_d_array = create_d_array(parameters, all_hkl_orig_unique)
    minimal = np.min(all_hkl_orig_unique_d_array)
    maximal = np.max(all_hkl_orig_unique_d_array)
    bins = int((maximal - minimal) / step)
    all_hist_orig, all_bin_edges_orig = np.histogram(all_hkl_orig_unique_d_array, bins)
    hist_orig, rest = np.histogram(hkl_orig_unique_d_array, bins=bins, range=(minimal, maximal))
    x_coord = (all_bin_edges_orig[0:-1] + all_bin_edges_orig[1:]) / 2
    hist_out = hist_orig / all_hist_orig * 100
    x_coord = x_coord[~np.isnan(hist_out)]
    hist_out = hist_out[~np.isnan(hist_out)]
    return hist_out, x_coord


def generate_multiplicity_plot(hkl_orig: np.ndarray, all_hkl_orig: np.ndarray, parameters: List[float],
                               step: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    d_hkl_orig = create_d_array(parameters, hkl_orig)
    d_all_hkl_orig = create_d_array(parameters, np.unique(all_hkl_orig, axis=0))
    minimal = np.min(d_all_hkl_orig)
    maximal = np.max(d_all_hkl_orig)
    bins = int((maximal - minimal) / step)
    all_hist, all_bin_edges = np.histogram(d_all_hkl_orig, bins)
    hist, bin_edges = np.histogram(d_hkl_orig, bins, range=(minimal, maximal))
    x_coord = (all_bin_edges[0:-1] + all_bin_edges[1:]) / 2
    hist_out = hist / all_hist
    x_coord = x_coord[~np.isnan(hist_out)]
    hist_out = hist_out[~np.isnan(hist_out)]
    return hist_out, x_coord


def generate_redundancy_plot(hkl: np.ndarray, all_hkl_orig: np.ndarray, parameters: List[float], step: float = 0.1) -> \
        Tuple[np.ndarray, np.ndarray]:
    d_hkl = create_d_array(parameters, hkl)
    d_all_hkl = create_d_array(parameters, all_hkl_orig)
    minimal = np.min(d_all_hkl)
    maximal = np.max(d_all_hkl)
    bins = int((maximal - minimal) / step)
    all_hist, all_bin_edges = np.histogram(d_all_hkl, bins)
    hist, bin_edges = np.histogram(d_hkl, bins, range=(minimal, maximal))
    x_coord = (all_bin_edges[0:-1] + all_bin_edges[1:]) / 2
    hist_out = hist / all_hist
    x_coord = x_coord[~np.isnan(hist_out)]
    hist_out = hist_out[~np.isnan(hist_out)]
    return hist_out, x_coord


def create_d_array(parameters: List[float], hkl_array: np.ndarray) -> np.ndarray:
    a, b, c, alpha_deg, beta_deg, gamma_deg = [float(p) for p in parameters]

    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)

    cos_a = np.cos(alpha)
    cos_b = np.cos(beta)
    cos_g = np.cos(gamma)
    sqrt_term = 1 - cos_a ** 2 - cos_b ** 2 - cos_g ** 2 + 2 * cos_a * cos_b * cos_g
    cell_vol = a * b * c * np.sqrt(sqrt_term)

    sin_a = np.sin(alpha)
    sin_b = np.sin(beta)
    sin_g = np.sin(gamma)

    h, k, l = hkl_array[:, 0], hkl_array[:, 1], hkl_array[:, 2]

    denominator = (
            h ** 2 * b ** 2 * c ** 2 * sin_a ** 2 +
            k ** 2 * a ** 2 * c ** 2 * sin_b ** 2 +
            l ** 2 * a ** 2 * b ** 2 * sin_g ** 2 +
            2 * h * k * a * b * c ** 2 * (cos_a * cos_b - cos_g) +
            2 * k * l * a ** 2 * b * c * (cos_b * cos_g - cos_a) +
            2 * h * l * a * b ** 2 * c * (cos_a * cos_g - cos_b)
    )

    d_array = cell_vol / np.sqrt(denominator)
    return d_array


def make_line_plot_graph(data: Tuple[np.ndarray, ...], all_hkl_orig: np.ndarray, parameters: List[float], step: float,
                         completeness: bool = True, redundancy: bool = True, multiplicity: bool = True) -> Tuple[
    Optional[go.Figure], Optional[go.Figure], Optional[go.Figure]]:
    comp_fig = None
    mult_fig = None
    red_fig = None
    if completeness is True:
        comp_y, comp_x = generate_completeness_plot(hkl_orig=data[1], all_hkl_orig=all_hkl_orig, parameters=parameters,
                                                    step=step)
        comp_df = {'x': comp_x, 'y': comp_y, 'color': 'blue'}
        comp_fig = px.line(comp_df, x='x', y='y')
        comp_fig['layout']['xaxis']['autorange'] = "reversed"
        comp_fig.update_traces(line_color='blue', line_width=3)
        comp_fig.update_xaxes({'title': 'd, Å'})
        comp_fig.update_yaxes({'title': '%'})

    if multiplicity is True:
        mult_y, mult_x = generate_multiplicity_plot(hkl_orig=data[1], all_hkl_orig=all_hkl_orig, parameters=parameters,
                                                    step=step)
        mult_df = {'x': mult_x, 'y': mult_y, 'color': 'green'}
        mult_fig = px.line(mult_df, x='x', y='y')
        mult_fig['layout']['xaxis']['autorange'] = "reversed"
        mult_fig.update_traces(line_color='green', line_width=3)
        mult_fig.update_xaxes({'title': 'd, Å'})
        mult_fig.update_yaxes({'title': ''})

    if redundancy is True:
        red_y, red_x = generate_redundancy_plot(hkl=data[0], all_hkl_orig=all_hkl_orig, parameters=parameters,
                                                step=step)
        red_df = {'x': red_x, 'y': red_y, 'color': 'red'}
        red_fig = px.line(red_df, x='x', y='y')
        red_fig['layout']['xaxis']['autorange'] = "reversed"
        red_fig.update_traces(line_color='red', line_width=3)
        red_fig.update_xaxes({'title': 'd, Å'})
        red_fig.update_yaxes({'title': ''})

    return comp_fig, mult_fig, red_fig


def ang_bw_two_vects(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray], type: str = 'list',
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


def decode_hkl(array: np.ndarray) -> np.ndarray:
    base = 1001
    l = np.round(array / base ** 2).astype(int)
    k = np.round((array - l * base ** 2) / base).astype(int)
    h = (array - l * base ** 2 - k * base).astype(int)
    decoded_array = np.hstack((h.reshape(-1, 1), k.reshape(-1, 1), l.reshape(-1, 1)))
    return decoded_array


class plotly_fig():
    def __init__(self, fig: go.Figure):
        self.fig = fig
        self.selected_points = None
        self.unknown_color = '#ff0000'
        self.selected_color = '#000000'

        self.n_traces = len(self.fig.data[:-3])
        self.active_points_i = [tuple() for i in range(self.n_traces)]
        self.colors = rgb_colors.colors[:self.n_traces - 2] + ('#ff0000', '#000000')

    def add_point_to_active(self, index_n: int, trace_n: int) -> None:
        if 'axis' not in self.fig.data[trace_n]['name']:
            self.active_points_i[trace_n] += (index_n,)
            self.active_points_i[trace_n] += tuple(set(self.active_points_i[trace_n]))
            self.fig.data[trace_n].marker.color = (self.fig.data[trace_n].marker.color[0:index_n] + ('#fd3db5',)
                                                   + self.fig.data[trace_n].marker.color[index_n + 1:])

    def to_active(self, trace_n: int, boolarray: np.ndarray) -> None:
        if boolarray.shape != (0,):
            colors = np.array(self.fig.data[trace_n]['marker']['color']).reshape(-1, 1)
            colors[boolarray] = '#fd3db5'

            self.fig.data[trace_n]['marker']['color'] = tuple(colors.reshape(-1))
            indices = tuple(*np.nonzero(boolarray))
            self.active_points_i[trace_n] = self.active_points_i[trace_n] + indices
            self.active_points_i[trace_n] = tuple(set(self.active_points_i[trace_n]))

    def delete_active(self) -> None:
        for trace_n in range(self.n_traces):
            if self.fig.data[trace_n]['x'] is not None:
                if self.fig.data[trace_n]['x'].shape != (0,):
                    if self.active_points_i[trace_n] != tuple():
                        xyz_array = np.vstack(
                            (self.fig.data[trace_n]['x'], self.fig.data[trace_n]['y'],
                             self.fig.data[trace_n]['z'])).transpose()
                        hkl_ohkl = self.fig.data[trace_n]['customdata']
                        xyz_hkl_ohkl = np.hstack((xyz_array, hkl_ohkl))
                        bool_array = np.ones(xyz_hkl_ohkl.shape[0]).astype(bool)
                        bool_array[np.array(self.active_points_i[trace_n])] = False
                        xyz_hkl_ohkl_new = xyz_hkl_ohkl[bool_array.reshape(-1, 1)[:, 0]]
                        color = tuple(np.array(self.fig.data[trace_n]['marker']['color'])[bool_array])
                        x = xyz_hkl_ohkl_new[:, 0].reshape(-1)
                        y = xyz_hkl_ohkl_new[:, 1].reshape(-1)
                        z = xyz_hkl_ohkl_new[:, 2].reshape(-1)
                        customdata = xyz_hkl_ohkl_new[:, 3:]
                        self.fig.data[trace_n]['customdata'] = customdata
                        self.fig.data[trace_n]['x'] = x
                        self.fig.data[trace_n]['y'] = y
                        self.fig.data[trace_n]['z'] = z
                        self.fig.data[trace_n]['marker']['color'] = color
                        self.active_points_i[trace_n] = tuple()

    def delete_selection_figure(self) -> None:
        for n, trace in enumerate(self.fig.data):
            if 'selection' in trace['name']:
                self.fig.data = self.fig.data[:n] + self.fig.data[n + 1:]
                break

    def create_selection_sphere(self) -> None:
        self.delete_selection_figure()
        r = 0.4
        points = 20
        theta = np.linspace(0, 2 * np.pi, points)
        phi = np.linspace(0, np.pi, points)
        theta, phi = np.meshgrid(theta, phi)

        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        surfacecolor = np.ones_like(z)
        self.fig.add_surface(x=x, y=y, z=z,
                             opacity=0.2,
                             hoverinfo='skip',
                             customdata=np.array([0, 0, 0, r]),
                             contours=go.surface.Contours(
                                 x=go.surface.contours.X(highlight=False),
                                 y=go.surface.contours.Y(highlight=False),
                                 z=go.surface.contours.Z(highlight=False),
                             ),
                             showscale=False,
                             name='selection sphere',
                             surfacecolor=surfacecolor
                             )

    def create_selection_cone(self) -> None:
        self.delete_selection_figure()
        theta = 30
        points = 50
        max_z = self.fig.data[self.n_traces]['x'][0] * 0.9
        max_xy = max_z * np.tan(np.deg2rad(theta)) / 2 ** 0.5
        lim_z = np.tan(np.deg2rad(theta)) ** -1 * (max_xy ** 2) ** 0.5
        x = np.linspace(-max_xy, max_xy, points)
        y = np.linspace(-max_xy, max_xy, points)
        x, y = np.meshgrid(x, y)
        z = np.tan(np.deg2rad(theta)) ** -1 * (x ** 2 + y ** 2) ** 0.5
        bool_arr = z > lim_z
        x[bool_arr] = np.nan
        y[bool_arr] = np.nan
        z[bool_arr] = np.nan

        self.fig.add_surface(x=x, y=y, z=z,
                             opacity=0.2,
                             hoverinfo='skip',
                             customdata=np.array([0, 0, 0, theta, 0, 0, 0], dtype='float32'),
                             contours=go.surface.Contours(
                                 x=go.surface.contours.X(highlight=False),
                                 y=go.surface.contours.Y(highlight=False),
                                 z=go.surface.contours.Z(highlight=False),
                             ),
                             showscale=False,
                             name='selection cone'
                             )

    def create_selection_cuboid(self) -> None:
        self.delete_selection_figure()
        a, b, c = (0.2, 0.2, 0.2)
        x = np.array([a, -a, a, -a, a, -a, a, -a, a, -a, -a, -a, -a, -a, -a, -a, -a, -a, a, a, a, a, a, a]).reshape(12,
                                                                                                                    2)
        y = np.array([b, b, -b, -b, -b, -b, b, b, b, b, b, b, b, -b, b, -b, -b, -b, b, b, b, -b, b, -b]).reshape(12, 2)
        z = np.array([-c, -c, -c, -c, c, c, c, c, -c, -c, -c, -c, -c, -c, c, c, c, c, c, c, c, c, -c, -c]).reshape(12,
                                                                                                                   2)
        self.fig.add_surface(x=x, y=y, z=z,
                             opacity=0.2,
                             hoverinfo='skip',
                             customdata=np.array([0, 0, 0, a, b, c, 0, 0, 0], dtype='float32'),
                             contours=go.surface.Contours(
                                 x=go.surface.contours.X(highlight=False),
                                 y=go.surface.contours.Y(highlight=False),
                                 z=go.surface.contours.Z(highlight=False),
                             ),
                             showscale=False,
                             name='selection cuboid'
                             )

    def resize_cuboid(self, axis: str, growth: float) -> None:
        zero = self.fig.data[-1]['customdata'][:3]
        abc_parameters = self.fig.data[-1]['customdata'][3:6]
        if axis == 'x':
            if abc_parameters[0] + growth > 0.05:
                abc_parameters[0] += growth
        elif axis == 'y':
            if abc_parameters[1] + growth > 0.05:
                abc_parameters[1] += growth
        elif axis == 'z':
            if abc_parameters[2] + growth > 0.05:
                abc_parameters[2] += growth
        self.fig.data[-1]['customdata'][3:6] = abc_parameters
        a, b, c = abc_parameters
        x = np.array([a, -a, a, -a, a, -a, a, -a, a, -a, -a, -a, -a, -a, -a, -a, -a, -a, a, a, a, a, a, a])
        y = np.array([b, b, -b, -b, -b, -b, b, b, b, b, b, b, b, -b, b, -b, -b, -b, b, b, b, -b, b, -b])
        z = np.array([-c, -c, -c, -c, c, c, c, c, -c, -c, -c, -c, -c, -c, c, c, c, c, c, c, c, c, -c, -c])
        xyz_array = np.vstack((x, y, z)).transpose()
        rot = R.from_euler('xyz', self.fig.data[-1]['customdata'][6:], degrees=True)
        xyz_array = rot.apply(xyz_array)
        self.fig.data[-1]['x'] = xyz_array[:, 0].reshape(12, 2) + zero[0]
        self.fig.data[-1]['y'] = xyz_array[:, 1].reshape(12, 2) + zero[1]
        self.fig.data[-1]['z'] = xyz_array[:, 2].reshape(12, 2) + zero[2]

    def move_selection_figure(self, axis: str, distance: float) -> None:
        if axis == 'x':
            self.fig.data[-1]['x'] = distance + self.fig.data[-1]['x']
            self.fig.data[-1]['customdata'][0] += distance
        elif axis == 'y':
            self.fig.data[-1]['y'] = distance + self.fig.data[-1]['y']
            self.fig.data[-1]['customdata'][1] += distance
        elif axis == 'z':
            self.fig.data[-1]['z'] = distance + self.fig.data[-1]['z']
            self.fig.data[-1]['customdata'][2] += distance

    def resize_selection_figure(self, axis: str, growth: float) -> None:
        if 'sphere' in self.fig.data[-1]['name']:
            self.resize_sphere(growth=growth)
        elif 'cone' in self.fig.data[-1]['name']:
            self.change_cone_angle(angle=growth * 100)
        elif 'cuboid' in self.fig.data[-1]['name']:
            self.resize_cuboid(axis=axis, growth=growth)

    def resize_sphere(self, growth: float) -> None:
        zero = self.fig.data[-1]['customdata'][:3]
        radii = self.fig.data[-1]['customdata'][3]
        multiplier = (radii + growth) / radii
        self.fig.data[-1]['x'] = (self.fig.data[-1]['x'] - zero[0]) * multiplier + zero[0]
        self.fig.data[-1]['y'] = (self.fig.data[-1]['y'] - zero[1]) * multiplier + zero[1]
        self.fig.data[-1]['z'] = (self.fig.data[-1]['z'] - zero[2]) * multiplier + zero[2]
        self.fig.data[-1]['customdata'][3] += growth

    def active_by_figure(self) -> None:
        if 'sphere' in self.fig.data[-1]['name']:
            self.active_by_sphere()
        elif 'cone' in self.fig.data[-1]['name']:
            self.active_by_cone()
        elif 'cuboid' in self.fig.data[-1]['name']:
            self.active_by_cuboid()

    def active_by_sphere(self) -> None:
        zero = self.fig.data[-1]['customdata'][:3]
        radii = self.fig.data[-1]['customdata'][3]
        for n, trace in enumerate(self.fig.data[:-4]):
            if trace['x'] is not None:
                if trace['x'].shape != (0,):
                    xyz_array = np.vstack((trace['x'], trace['y'], trace['z'])).transpose()
                    xyz_array[:, 0] -= zero[0]
                    xyz_array[:, 1] -= zero[1]
                    xyz_array[:, 2] -= zero[2]
                    bool_array = ((xyz_array[:, 0] ** 2 + xyz_array[:, 1] ** 2 + xyz_array[:, 2] ** 2) ** 0.5) < radii
                    self.to_active(trace_n=n, boolarray=bool_array)

    def active_by_cone(self) -> None:
        rotation = R.from_euler('xyz', self.fig.data[-1]['customdata'][4:], degrees=True)
        zero = self.fig.data[-1]['customdata'][:3]
        height = rotation.apply(np.array([0, 0, 1]))
        theta_cos = np.cos(np.deg2rad(self.fig.data[-1]['customdata'][3]))
        for n, trace in enumerate(self.fig.data[:-4]):
            if trace['x'] is not None:
                if trace['x'].shape != (0,):
                    xyz_array = np.vstack((trace['x'], trace['y'], trace['z'])).transpose()
                    xyz_array[:, 0] -= zero[0]
                    xyz_array[:, 1] -= zero[1]
                    xyz_array[:, 2] -= zero[2]
                    cos_array = ang_bw_two_vects(xyz_array, height, 'array', 'cos')
                    bool_array = cos_array > theta_cos
                    self.to_active(trace_n=n, boolarray=bool_array)

    def active_by_cuboid(self) -> None:
        rotation = R.from_euler('xyz', self.fig.data[-1]['customdata'][6:], degrees=True)
        zero = self.fig.data[-1]['customdata'][:3]
        abc_parameters = self.fig.data[-1]['customdata'][3:6]
        xyz_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        xyz_vecs = rotation.apply(xyz_vecs)
        for n, trace in enumerate(self.fig.data[:-4]):
            if trace['x'] is not None:
                if trace['x'].shape != (0,):
                    xyz_array = np.vstack((trace['x'], trace['y'], trace['z'])).transpose()
                    xyz_array[:, 0] -= zero[0]
                    xyz_array[:, 1] -= zero[1]
                    xyz_array[:, 2] -= zero[2]
                    proj_x = np.dot(xyz_array, xyz_vecs[0, :])
                    proj_y = np.dot(xyz_array, xyz_vecs[1, :])
                    proj_z = np.dot(xyz_array, xyz_vecs[2, :])
                    bool_array_x = np.abs(proj_x) < abc_parameters[0]
                    bool_array_y = np.abs(proj_y) < abc_parameters[1]
                    bool_array_z = np.abs(proj_z) < abc_parameters[2]
                    bool_array = bool_array_x & bool_array_y & bool_array_z

                    self.to_active(trace_n=n, boolarray=bool_array)

    def rotate_sel_figure(self, axis: str, angle: float) -> None:
        if 'sphere' in self.fig.data[-1]['name']:
            pass
        elif 'cone' in self.fig.data[-1]['name'] or 'cuboid' in self.fig.data[-1]['name']:
            self.rotate_fig(axis, angle)

    def change_cone_angle(self, angle: float) -> None:
        theta = self.fig.data[-1]['customdata'][3] + angle
        if theta <= 89 and theta >= 1:
            self.fig.data[-1]['customdata'][3] = theta
            points = 50
            max_z = self.fig.data[self.n_traces]['x'][0] * 0.9
            max_xy = max_z * np.tan(np.deg2rad(theta)) / 2 ** 0.5
            lim_z = np.tan(np.deg2rad(theta)) ** -1 * (max_xy ** 2) ** 0.5
            x = np.linspace(-max_xy, max_xy, points)
            y = np.linspace(-max_xy, max_xy, points)
            x, y = np.meshgrid(x, y)
            z = np.tan(np.deg2rad(theta)) ** -1 * (x ** 2 + y ** 2) ** 0.5
            shape = z.shape
            bool_arr = z > lim_z
            x[bool_arr] = np.nan
            y[bool_arr] = np.nan
            z[bool_arr] = np.nan
            xyz_array = np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1))).transpose()
            rot = R.from_euler('xyz', self.fig.data[-1]['customdata'][4:], degrees=True)
            xyz_array = rot.apply(xyz_array)
            zero = self.fig.data[-1]['customdata'][:3]
            xyz_array[:, 0] += zero[0]
            xyz_array[:, 1] += zero[1]
            xyz_array[:, 2] += zero[2]
            self.fig.data[-1]['x'] = xyz_array[:, 0].reshape(shape)
            self.fig.data[-1]['y'] = xyz_array[:, 1].reshape(shape)
            self.fig.data[-1]['z'] = xyz_array[:, 2].reshape(shape)

    def rotate_fig(self, axis: str, angle: float) -> None:
        if 'cone' in self.fig.data[-1]['name']:
            rot_initial = R.from_euler('xyz', self.fig.data[-1]['customdata'][4:], degrees=True)
        else:
            rot_initial = R.from_euler('xyz', self.fig.data[-1]['customdata'][6:], degrees=True)
        rot_new = R.from_euler(axis, angle, degrees=True)
        xyz_rot = (rot_new * rot_initial).as_euler('xyz', degrees=True).astype('float32')
        zero = self.fig.data[-1]['customdata'][:3]
        shape = self.fig.data[-1]['x'].shape
        xyz_array = np.vstack(((self.fig.data[-1]['x'] - zero[0]).reshape(-1),
                               (self.fig.data[-1]['y'] - zero[1]).reshape(-1),
                               (self.fig.data[-1]['z'] - zero[2]).reshape(-1))).transpose()
        xyz_array = rot_new.apply(xyz_array)
        xyz_array[:, 0] += zero[0]
        xyz_array[:, 1] += zero[1]
        xyz_array[:, 2] += zero[2]
        self.fig.data[-1]['x'] = xyz_array[:, 0].reshape(shape)
        self.fig.data[-1]['y'] = xyz_array[:, 1].reshape(shape)
        self.fig.data[-1]['z'] = xyz_array[:, 2].reshape(shape)
        if 'cone' in self.fig.data[-1]['name']:
            self.fig.data[-1]['customdata'][4:] = xyz_rot
        else:
            self.fig.data[-1]['customdata'][6:] = xyz_rot

    def reset_active(self) -> None:
        runs = 0
        for trace in self.fig.data:
            if 'run' in trace['name']:
                runs += 1
        if self.active_points_i != [tuple() for i in range(runs + 2)]:
            for trace_n in range(runs):
                if self.active_points_i[trace_n] != tuple():
                    self.active_points_i[trace_n] = tuple()

                    colors = np.array(self.fig.data[trace_n]['marker']['color']).reshape(-1, 1)
                    colors[:] = rgb_colors.colors[trace_n]
                    self.fig.data[trace_n]['marker']['color'] = tuple(colors.reshape(-1))
            if self.active_points_i[runs] != tuple():
                self.active_points_i[runs] = tuple()
                colors = np.array(self.fig.data[runs]['marker']['color']).reshape(-1, 1)
                colors[:] = self.unknown_color
                self.fig.data[runs]['marker']['color'] = tuple(colors.reshape(-1))

            if self.active_points_i[runs + 1] != tuple():
                self.active_points_i[runs + 1] = tuple()
                colors = np.array(self.fig.data[runs + 1]['marker']['color']).reshape(-1, 1)
                colors[:] = self.selected_color
                self.fig.data[runs + 1]['marker']['color'] = tuple(colors.reshape(-1))

    def invert_active(self) -> None:
        for trace_n in range(self.n_traces):
            if self.fig.data[trace_n]['x'] is not None:
                if self.fig.data[trace_n]['x'].shape != (0,):
                    if self.active_points_i[trace_n] != tuple():
                        shape = len(self.fig.data[trace_n]['x'])
                        bool_array = np.ones(shape, dtype=bool)
                        bool_array[np.array(self.active_points_i[trace_n])] = False
                        colors = (self.colors[trace_n],) * shape
                        self.fig.data[trace_n]['marker']['color'] = colors
                        self.active_points_i[trace_n] = tuple()
                        self.to_active(trace_n=trace_n, boolarray=bool_array)
                    else:
                        shape = len(self.fig.data[trace_n]['x'])
                        bool_array = np.ones(shape, dtype=bool)
                        self.to_active(trace_n=trace_n, boolarray=bool_array)

    def unknown_to_active(self) -> None:
        if self.fig.data[self.n_traces - 2]['x'] is not None:
            if self.fig.data[self.n_traces - 2]['x'].shape != (0,):
                shape = self.fig.data[self.n_traces - 2]['x'].shape
                bool_array = np.ones(shape, dtype=bool)
                self.to_active(trace_n=self.n_traces - 2, boolarray=bool_array)

    def active_to_selected(self) -> None:
        x = list()
        y = list()
        z = list()
        custom_data = list()
        for trace_n in range(self.n_traces):
            if self.active_points_i[trace_n] != tuple():
                shape = len(self.fig.data[trace_n]['x'])
                bool_array = np.zeros(shape, dtype=bool)
                bool_array[np.array(self.active_points_i[trace_n])] = True
                x += (self.fig.data[trace_n]['x'][bool_array],)
                y += (self.fig.data[trace_n]['y'][bool_array],)
                z += (self.fig.data[trace_n]['z'][bool_array],)
                custom_data += (self.fig.data[trace_n]['customdata'][bool_array.reshape(-1, 1)[:, 0]],)
            else:
                x += [np.array([]), ]
                y += [np.array([]), ]
                z += [np.array([]), ]
                custom_data += [np.array([[]]).reshape(0, 6), ]
        self.delete_active()
        x_selected = self.fig.data[self.n_traces - 1]['x']
        y_selected = self.fig.data[self.n_traces - 1]['y']
        z_selected = self.fig.data[self.n_traces - 1]['z']
        hkl_hkl_o_selected = self.fig.data[self.n_traces - 1]['customdata']

        x_new = np.hstack((x_selected, *x))
        y_new = np.hstack((y_selected, *y))
        z_new = np.hstack((z_selected, *z))

        hkl_hkl_o_new = np.vstack((hkl_hkl_o_selected, *custom_data))

        hkl_array = hkl_hkl_o_new[:, 0:3]
        hkl, indices = np.unique(hkl_array, axis=0, return_index=True)
        x_new = x_new[indices]
        y_new = y_new[indices]
        z_new = z_new[indices]
        hkl_hkl_o_new = hkl_hkl_o_new[indices]
        shape = len(x_new)
        self.fig.data[self.n_traces - 1]['x'] = x_new
        self.fig.data[self.n_traces - 1]['y'] = y_new
        self.fig.data[self.n_traces - 1]['z'] = z_new
        self.fig.data[self.n_traces - 1]['customdata'] = hkl_hkl_o_new
        self.fig.data[self.n_traces - 1]['marker']['color'] = (self.colors[-1],) * shape


def parse_logic_exp_txt(exp_str: str, block_sep: str = ';', subblock_sep: str = '\n', element_sep: str = ',') -> List[
    List[List[str]]]:
    blocks = exp_str.split(block_sep)
    subblocks = [[j.split(element_sep) for j in i if j != ''] for i in [block.split(subblock_sep) for block in blocks]]
    subblocks = [block for block in subblocks if block != list()]
    return subblocks


@mylogger(log_args=True)
def logic_eval(string_expression: str, variables: Dict[str, Any] = {}) -> bool:
    code = compile(string_expression, "<string>", "eval")
    for name in code.co_names:
        if name not in variables:
            raise NameError(f"Usage of {name} is not allowed.")
    result = eval(code, {"__builtins__": {}}, variables)
    if (isinstance(result, (bool, np.bool_)) or
            (isinstance(result, np.ndarray) and result.dtype == bool)):
        return result
    else:
        raise TypeError(f"Result of expression should be bool or logic ndarray, not {type(result)}")


def parse_p4p_for_UB(str_data: str) -> Optional[np.ndarray]:
    try:
        lines_list = str_data.split('\n')
        lines_orts = [i.split() for i in lines_list if 'ORT' in i]
        ub_components = [[float(i[1]), float(i[2]), float(i[3])] for i in lines_orts]
        ub_matrix = np.array(ub_components)
        return ub_matrix
    except:
        return None


def parse_par_for_UB(str_data: str, to_angstroms: bool = True) -> Optional[np.ndarray]:
    try:
        lines_list = str_data.split('\n')
        CRYSALIS_WAVELENGTHS = [
            {'expression': '   - WAVELENGTH ', 'exception': None, 'index': 6},
        ]
        for entry in CRYSALIS_WAVELENGTHS:
            for line in lines_list:
                if entry['expression'] in line:
                    if entry['exception'] and entry['exception'] in line:
                        continue
                    parts = line.split()
                    try:
                        wavelength = float(parts[entry['index']])
                        print(f"Found wavelength: {wavelength}")
                        break
                    except (IndexError, ValueError):
                        continue
                else:
                    continue

        ub_line = next(i for i in lines_list if 'CRYSTALLOGRAPHY UB' in i and not '§' in i).split()
        ub_components = ub_line[2:]
        ub_components = [float(i) for i in ub_components]
        ub_matrix = np.array(ub_components).reshape(3, 3)
        if to_angstroms: ub_matrix /= wavelength
        esp_to_brk_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        ub_matrix = np.matmul(esp_to_brk_rotation, ub_matrix)
        return ub_matrix
    except:
        return None


def form_data_for_gon_table(axes_data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    table_template = {'rotation': '', 'direction': '', 'name': '', 'real': '', 'angle': 0}
    output_data = []
    for i in range(0, len(list(axes_data.values())[0])):
        axes_data['axes_directions'][i] = '+' if axes_data['axes_directions'][i] == 1 else '-'
        process_dict = dict({'name_col': axes_data['axes_names'][i], 'rotation_col': axes_data['axes_rotations'][i],
                             'direction_col': axes_data['axes_directions'][i], 'real_col': axes_data['axes_real'][i],
                             'ang_col': axes_data['axes_angles'][i], **table_template})
        output_data.append(process_dict)
    return output_data


def get_float_from_str(str_: str, all_: bool = False) -> Union[float, List[float], None]:
    float_pattern = r"[-+]?\d*\.\d+|\d+\.\d*"
    if all_:
        float_list = re.findall(float_pattern, str_)
        float_list = [float(float_) for float_ in float_list]
        return float_list
    else:
        if re.search(float_pattern, str_):
            return float(re.search(float_pattern, str_)[0])
        else:
            return None


def process_dcc_upload_json(contents: str) -> Dict[str, Any]:
    decoded_data = base64.b64decode(contents.split(',')[1])
    try:
        data_str = decoded_data.decode('utf-8')
    except:
        data_str = decoded_data.decode('latin-1')
    data_dict = json.loads(data_str)
    return data_dict


def process_dcc_upload_file_to_str(contents: str) -> str:
    decoded_data = base64.b64decode(contents.split(',')[1])
    try:
        data_str = decoded_data.decode('utf-8')
    except:
        data_str = decoded_data.decode('latin-1')
    data_str = data_str.replace('\r', '')
    return data_str


def generate_str_for_det_download(experiment_instance: Any) -> str:
    if not isinstance(experiment_instance, Experiment):
        raise TypeError(f'{experiment_instance} is not an instance of {Experiment}')
    det_geometry = experiment_instance.det_geometry
    height, width = experiment_instance.det_height, experiment_instance.det_width

    if det_geometry == 'circle':
        diameter = experiment_instance.det_diameter
        str_ = f'det_geom={det_geometry}\ndiameter={diameter}'
    else:
        complex_det = experiment_instance.det_complex
        if not complex_det:
            str_ = f'det_geom={det_geometry}\nheight={height}\nwidth={width}\ncomplex={complex_det}'
        else:
            rows, cols = experiment_instance.det_complex_format
            row_spacing, col_spacing = experiment_instance.det_row_col_spacing
            str_ = (
                f'det_geom={det_geometry}\nheight={height}\nwidth={width}\ncomplex={complex_det}\nrows={rows}\ncols={cols}\nrow_spacing={row_spacing}\ncol_spacing={col_spacing}')
    return str_


def check_detector_dict(dict_: Dict[str, Any]) -> bool:
    if not dict_has_keys(dict_, ['det_geometry'], exact_keys=False): return False
    if dict_['det_geometry'] == 'circle':
        if not dict_has_keys(dict_, ['det_diameter'], exact_keys=False): return False
        if check_dict_value(dict_, 'det_diameter', (int, float), higher=0):
            return True
        else:
            return False

    elif dict_['det_geometry'] == 'rectangle':
        if not dict_has_keys(dict_, ['det_width', 'det_height', 'det_complex'], exact_keys=False): return False

        if not all((check_dict_value(dict_, 'det_height', (int, float), higher=0),
                    check_dict_value(dict_, 'det_width', (int, float), higher=0),
                    check_dict_value(dict_, 'det_complex', (bool,)))): return False
        if not dict_.get('det_complex', None):
            return True
        else:
            try:
                if not dict_has_keys(dict_, ['det_complex_format', 'det_row_col_spacing'], exact_keys=False):
                    return False
                bool_ = all((len(dict_['det_complex_format']) == 2, len(dict_['det_row_col_spacing']) == 2,
                             dict_['det_complex_format'][0] > 0, dict_['det_complex_format'][1] > 0,
                             type(dict_['det_complex_format'][0]) is int, type(dict_['det_complex_format'][1]) is int,
                             dict_['det_row_col_spacing'][0] > 0, dict_['det_row_col_spacing'][1] > 0))

            except:
                return False
            if bool_:
                return True
            else:
                return False
    else:
        return False


def parse_text_to_dict(text: str, delimiter: str) -> Dict[str, Union[str, int, float]]:
    pairs = text.split(delimiter)
    result = {}

    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        result[key.strip()] = value
    return result


def check_dict_value(dict_: Dict[str, Any], keys: Union[str, List[str]], classes: Union[type, List[type], tuple],
                     check_pos_value: bool = False, lower: Optional[float] = None,
                     higher: Optional[float] = None, equal_to: Optional[Union[Any, List[Any]]] = None) -> bool:
    ...
    if not (type(classes) in [list, tuple]):
        classes = [classes, ]
    if type(keys) == str: keys = [keys, ]
    for key in keys:
        if not (key in dict_):
            return False
        if not (type(dict_[key]) in classes):
            return False
        if check_pos_value and dict_[key] <= 0:
            return False
        if lower:
            if not (dict_[key] < lower):
                return False
        if higher:
            if not (dict_[key] > higher):
                return False
        if equal_to:
            if not (type(equal_to) is list) or (not type(equal_to) is tuple):
                equal_to = [equal_to, ]
            if not any([dict_[key] == i for i in equal_to]):
                return False
    return True


def generate_data_for_detector_table(diameter: Optional[float] = None, height: Optional[float] = None,
                                     width: Optional[float] = None) -> List[Dict[str, Any]]:
    if diameter:
        data = [{'diameter_prm': diameter}, ]
    else:
        data = pd.DataFrame({'height_prm': [height, ],
                             'width_prm': [width, ]
                             }).to_dict('records')
    return data


def generate_dicts_for_obst(exp_inst: Any) -> List[Dict[str, Any]]:
    if not isinstance(exp_inst, Experiment):
        raise TypeError(f'{exp_inst} variable is not an instance of {Experiment}')
    dummy = [[*[None, ] * 8, [None, None, None], None], ]
    obstacle_list = exp_inst.obstacles if exp_inst.obstacles else dummy
    obstacle_parameters = ['distance', 'geometry', 'orientation', 'displacement_y', 'displacement_z', 'height', 'width',
                           'diameter', 'rotation_x', 'rotation_y', 'rotation_z', 'name']
    obst_dicts = []
    for obst in obstacle_list:
        obst = [*obst[:8], *obst[8], obst[9]]
        obstacle_dict = dict(zip(obstacle_parameters, obst))
        obst_dicts.append(obstacle_dict)
    return obst_dicts

def generate_dicts_for_linked_obst(exp_inst: Any) -> List[Dict[str, Any]]:
    if not isinstance(exp_inst, Experiment):
        raise TypeError(f'{exp_inst} variable is not an instance of {Experiment}')
    dummy = [[*[None, ] * 9, [None, None, None], None], ]
    obstacle_list = exp_inst.linked_obstacles if exp_inst.linked_obstacles else dummy
    obstacle_parameters = ['distance', 'geometry', 'orientation', 'displacement_y', 'displacement_z', 'height', 'width',
                           'diameter','linked_axis', 'rotation_x', 'rotation_y', 'rotation_z', 'name']
    obst_dicts = []
    for obst in obstacle_list:
        assert isinstance(obst, LinkedObstacle)
        obst = [obst.dist,obst.geometry,obst.orientation,obst.disp_y,obst.disp_z, obst.height,obst.width, obst.diameter,
                obst.highest_linked_axis_index,*obst.rot,obst.name]
        obstacle_dict = dict(zip(obstacle_parameters, obst))
        obst_dicts.append(obstacle_dict)
    return obst_dicts


def check_goniometer_dict(dict_: Dict[str, Any]) -> bool:
    if not dict_has_keys(dict_, ['axes_names', 'axes_directions', 'axes_rotations', 'axes_angles', 'axes_real']):
        return False
    if not (type(dict_['axes_names']) == str): return False
    if not (dict_['axes_directions'] in ['+', '-']): return False
    if not (dict_['axes_rotations'] in ['x', 'y', 'z']): return False
    if not (type(dict_['axes_angles']) in [float, int]): return False
    if not (dict_['axes_real'] in ['false', 'true']): return False
    return True


def generate_dicts_for_goniometer(exp_inst: Any) -> List[Dict[str, Any]]:
    if not isinstance(exp_inst, Experiment):
        raise TypeError(f'{exp_inst} variable is not an instance of {Experiment}')
    axes_list = []
    try:
        for name, rotation, direction, angles, real in zip(exp_inst.axes_names, exp_inst.axes_rotations,
                                                           exp_inst.axes_directions,
                                                           exp_inst.axes_angles, exp_inst.axes_real):
            axes_dict = {
                'axes_names': name,
                'axes_directions': '+' if direction == 1 else '-',
                'axes_rotations': rotation,
                'axes_angles': float(angles),
                'axes_real': real,
            }
            axes_list.append(axes_dict)
    except TypeError:
        axes_dict = {
            'axes_names': None,
            'axes_directions': None,
            'axes_rotations': None,
            'axes_angles': None,
            'axes_real': None,
        }
        axes_list.append(axes_dict)
    return axes_list


def generate_det_dict(exp_inst: Any) -> Dict[str, Any]:
    if not isinstance(exp_inst, Experiment):
        raise TypeError(f'{exp_inst} variable is not an instance of {Experiment}')
    detector_dict = {
        'det_geometry': exp_inst.det_geometry,
        'det_complex': exp_inst.det_complex
    }
    if exp_inst.det_height is not None and exp_inst.det_width is not None:
        detector_dict['det_height'] = float(exp_inst.det_height)
        detector_dict['det_width'] = float(exp_inst.det_width)
    if exp_inst.det_diameter is not None:
        detector_dict['det_diameter'] = float(exp_inst.det_diameter)
    if exp_inst.det_complex_format is not None:
        detector_dict['det_complex_format'] = [int(i) for i in exp_inst.det_complex_format]
        detector_dict['det_row_col_spacing'] = [float(i) for i in exp_inst.det_row_col_spacing]
    return detector_dict


def generate_scans_dicts_list(exp_inst: Any) -> List[Dict[str, Any]]:
    if not isinstance(exp_inst, Experiment):
        raise TypeError(f'{exp_inst} variable is not an instance of {Experiment}')
    runs_list = []
    for run in exp_inst.scans:
        scan_dict = {'scan_n': run[4],
                     'sweep': run[5]}
        detector_dict = {
            'det_dist': run[0],
            'det_orientation': run[2],
            'det_angles': {
                'x': run[1][0],
                'y': run[1][1],
                'z': run[1][2],
            },
            'det_disp_y': run[6],
            'det_disp_z': run[7],
        }
        axes_list = []
        for num, (angle, axis_name) in enumerate(zip(run[3], exp_inst.axes_names)):
            axes_dict = {'number': num,
                         'angle': angle,
                         'name': axis_name}
            axes_list.append(axes_dict)
        scan_dict['axes'] = axes_list
        scan_dict['detector'] = detector_dict
        runs_list.append(scan_dict)
    return runs_list


@mylogger(log_args=True, log_result=True)
def extract_run_data_into_list(dict_: Dict[str, Any]) -> Dict[str, Any]:
    angles = []
    order = []
    for axis in dict_['axes']:
        angles.append(axis['angle'])
        order.append(axis['number'])
    axes_angles = [angles[i] for i in order]
    run = {
        'det_dist': dict_['detector']['det_dist'],
        'det_angles': (
            dict_['detector']['det_angles']['x'], dict_['detector']['det_angles']['y'],
            dict_['detector']['det_angles']['z']),
        'det_orientation': dict_['detector']['det_orientation'],
        'axes_angles': axes_angles,
        'scan': dict_['scan_n'],
        'sweep': dict_['sweep'],
        'det_disp_y': dict_['detector']['det_disp_y'],
        'det_disp_z': dict_['detector']['det_disp_z'],
    }
    return run


@mylogger(log_args=True)
def process_runs_to_dicts_list(runs: List[List[Any]]) -> List[Dict[str, Any]]:
    data_ = []
    for run in runs:
        axes_angles = [i for i in run[9:]]
        dict_ = {'det_dist': run[1], 'det_angles': (run[3], run[4], run[5]), 'det_orientation': run[2],
                 'axes_angles': axes_angles,
                 'scan': int(run[0]), 'sweep': run[8], 'det_disp_y': run[6], 'det_disp_z': run[7]}
        data_.append(dict_)
    return data_


def check_run_dict_temp(run: Dict[str, Any]) -> None:
    if any((run['det_dist'] == '', run['det_angles'][0] == '', run['det_angles'][1] == '', run['det_angles'][2] == ''
            , run['det_orientation'] == '', run['scan'] == '', run['scan'] == '', run['sweep'] == '',
            run['sweep'] == 0)):
        raise RunsDictError(add_runs_empty_error)
    if run['det_orientation'] == 'independent' and (run['det_disp_y'] == '' or run['det_disp_z'] == ''):
        raise RunsDictError(add_runs_empty_error)
    if any([angle == '' for angle in run['axes_angles']]):
        raise RunsDictError(add_runs_empty_gon_error)


@mylogger(log_args=True)
def check_collision(exp_inst: Any, runs: List[Dict[str, Any]]) -> None:
    if not isinstance(exp_inst, Experiment):
        raise TypeError(f'{exp_inst} variable is not an instance of {Experiment}')
    runs_to_check = []
    for run in runs:
        runs_to_check.append([run['det_dist'], run['det_angles'], run['det_orientation'], run['axes_angles']
                                 , run['scan'], run['sweep'], run['det_disp_y'], run['det_disp_z']])
    exp_inst.check_collision_v2(scans=runs_to_check)


@mylogger(log_args=True)
def check_run_dict(exp_inst, dict_):
    if not isinstance(exp_inst, Experiment):
        raise TypeError(f'{exp_inst} variable is not an instance of {Experiment}')
    if len(dict_['axes']) != len(exp_inst.axes_rotations):
        raise RunsDictError(load_runs_axes_number_error)
    axes_angles = exp_inst.axes_angles
    axes_real = exp_inst.axes_real
    axes_nums = []

    for num, axis in enumerate(dict_['axes']):
        if axis['angle'] != axes_angles[axis['number']] and axes_real[axis['number']] == 'false':
            raise RunsDictError(load_runs_axes_angle_error)
        print(axis['number'])
        axes_nums.append(axis['number'])
    axes_nums.sort()
    if len(axes_nums) != len(set(axes_nums)):
        raise RunsDictError(check_dict_axes_repeating)
    if exp_inst.axes_real[dict_['scan_n']] == 'false':
        raise RunsDictError(check_dict_error_fake_scan)
    if len(dict_['axes']) != len(exp_inst.axes_rotations):
        raise RunsDictError(check_dict_axes_n_mismatch)
    if axes_nums != list(range(0, len(axes_nums))):
        raise RunsDictError(check_dict_axes_out_of_range)
    return True


def generate_data_for_runs_table(data: List[Dict[str, Any]], table_num: int) -> List[Dict[str, Any]]:
    data_list = []
    for run in data:
        dict_ = {
            f'{table_num}scan_no': run['scan_n'],
            f'{table_num}det_dist': run['detector']['det_dist'],
            f'{table_num}det_orientation': run['detector']['det_orientation'],
            f'{table_num}det_rot_x': run['detector']['det_angles']['x'],
            f'{table_num}det_rot_y': run['detector']['det_angles']['y'],
            f'{table_num}det_rot_z': run['detector']['det_angles']['z'],
            f'{table_num}det_disp_y': run['detector']['det_angles'],
            f'{table_num}det_disp_z': run['detector']['det_angles'],
            f'{table_num}scan_sweep': run['sweep'],
        }
        for axis in run['axes']:
            dict_[f"{table_num}_{axis['number']}_rot"] = axis['angle']
        table_num += 1
        data_list.append(dict_)
    return data_list


def write_json(to_save: Any) -> None:
    with open('obst.json', 'w', encoding='utf-8') as f:
        json.dump(to_save, f, ensure_ascii=False, indent=4)


def remake_obstacle_dict(dict_: Dict[str, Any]) -> Dict[str, Any]:
    keys_ = list(dict_.keys())
    i = find_index(keys_[0], '_', 3)
    keys = [key[i + 1:] for key in keys_]
    new_dict = dict(zip(keys, dict_.values()))
    return new_dict


def find_index(s: str, x: str, n: int) -> Optional[int]:
    count = 0
    for i, char in enumerate(s):
        if char == x:
            count += 1
            if count == n: return i
    return None


def check_obstacle_dict(dict_: Dict[str, Any], linked=False) -> bool:
    if not (check_dict_value(dict_, ['geometry', 'orientation'], classes=str, check_pos_value=False)
            and check_dict_value(dict_, ['distance', 'rotation_y', 'rotation_x', 'rotation_z'], check_pos_value=False,
                                 classes=(int, float))): return False
    if dict_['geometry'] == '' or dict_['orientation'] == '':
        return False
    if dict_['orientation'] == 'independent':
        if not check_dict_value(dict_, ['displacement_y', 'displacement_z'], classes=(float, int),
                                check_pos_value=False):
            return False
    if dict_['geometry'] == 'circle':
        if not check_dict_value(dict_, 'diameter', classes=(float, int)):
            return False
    elif dict_['geometry'] == 'rectangle':
        if not check_dict_value(dict_, ['width', 'height'], classes=(float, int)):
            return False
    if linked:
        if not check_dict_value(dict_,['linked_axis'],classes=int):
            return False
    return True


def dict_has_keys(dict_: Dict[str, Any], keys: List[str], exact_keys: bool = True) -> bool:
    if exact_keys:
        if set(dict_.keys()) == set(keys):
            return True
        else:
            return False
    else:
        if set(keys).issubset(set(dict_.keys())):
            return True
        else:
            return False


def list_of_dicts_to_dict(list_: List[Dict[str, Any]], merge_type: Dict[str, type]) -> Dict[str, Any]:
    dict_ = {}
    for key in merge_type.keys():
        if merge_type[key] == str:
            val = ''
            for dict_i in list_:
                val += dict_i[key]
        if merge_type[key] == list:
            val = []
            for dict_i in list_:
                val.append(dict_i[key])
        dict_[key] = val
    return dict_


def get_loaded_flags(data_loaded: Dict[str, Union[bool, None]]) -> Dict[str, Union[bool, None]]:
    flags = {key: data_loaded[key] if data_loaded[key] in (None, False) else True for key in data_loaded.keys()}
    flags = {key: None for key in ['wavelength', 'detector', 'obstacles', 'goniometer'] if not (key in flags.keys())}
    return flags


def procces_anvil_normal_data(data):
    normal = np.array([data[0]['anvil_normal_x'], data[0]['anvil_normal_y'], data[0]['anvil_normal_z']]).astype(
        np.float64)
    return normal


class CumulativeDataCalculator:
    data_container = {}
    high_free_index = 0
    d_low = None
    d_high = None
    hkl_all_e = None
    hkl_orig_all_e = None
    d_spacings_all = None
    cell_parameters = None

    def set_d_roi(self, roi):
        if roi is None:
            self.d_low = None
            self.d_high = None
        else:
            self.d_low = min(roi)
            self.d_high = max(roi)

    def create_spacings_for_hkl_e(self, hkl_e):
        d_array = create_d_array(parameters=self.cell_parameters, hkl_array=decode_hkl(hkl_e))
        return d_array

    def add_data(self, scandatacontainer):
        assert isinstance(scandatacontainer, ScanDataContainer), 'DiffractionData wrong format'
        self.data_container[self.high_free_index] = {
            'diff_vecs': scandatacontainer.diff_vecs,
            'diff_angles': scandatacontainer.diff_angles,
            'init_angles': scandatacontainer.start_angle,
            'sweep': scandatacontainer.sweep,
            'hkl_e': encode_hkl(scandatacontainer.hkl),
            'hkl_origin_e': encode_hkl(scandatacontainer.hkl_origin),
            'ommited': False,
            'order': self.high_free_index,
            'angle_roi': None,
            'd_spacings': create_d_array(parameters=self.cell_parameters, hkl_array=scandatacontainer.hkl).reshape(-1)
        }

        self.high_free_index += 1

    def clear_data(self):
        self.data_container = {}
        self.high_free_index = 0
        self.hkl_all_e = None
        self.hkl_orig_all_e = None
        self.cell_parameters = None
        self.d_spacings_all = None

    def get_data_for_cumulative_plot(self):
        hkl_original_e_list = []
        for run_data in self.data_container:
            bool_mask = self.gen_bool_mask(run_data)
            hkl_original_ = run_data['hkl_origin_e'][bool_mask].reshape(-1)
            hkl_original_e_list.append(hkl_original_)
        bool_mask = self.gen_d_roi_bool_mask(self.d_spacings_all)
        all_origin_e = self.hkl_orig_all_e[bool_mask].reshape(-1)
        return hkl_original_e_list, all_origin_e

    def change_ommit_by_index(self, index, flag):
        assert index in tuple(self.data_container.keys()), f'There is no data with {index} index!'
        self.data_container[index]['ommited'] = flag

    def del_data_by_index(self, index: int):
        r = self.data_container.pop(index, None)
        if r:
            if self.high_free_index - 1 == index:
                key_mapping = {old_i: old_i - 1 for old_i in self.data_container.keys() if index < old_i}
                for old_key, new_key in key_mapping.items():
                    self.data_container[new_key] = self.data_container.pop(old_key)
            self.high_free_index -= 1

    def add_all_hkl(self, hkl_all, hkl_orig_all):
        data = (hkl_all, hkl_orig_all)
        assert all(isinstance(hkl, np.ndarray) for hkl in data) and all(
            len(hkl[0]) == 3 for hkl in data), 'wrong hkl data format'
        self.hkl_all_e = encode_hkl(hkl_all)
        self.hkl_orig_all_e = encode_hkl(hkl_orig_all)
        self.d_spacings_all = self.create_spacings_for_hkl_e(self.hkl_all_e)

    def gen_d_roi_bool_mask(self, d_spacings):
        d_low_bool = d_spacings > self.d_low if self.d_low else True
        d_high_bool = d_spacings < self.d_high if self.d_high else True
        d_bool = d_low_bool & d_high_bool
        return d_bool

    def gen_angle_roi_bool_mask(self, angle_roi, diff_angles):
        if angle_roi:
            min_, max_, range_ = Sample.angle_range(start_rad=angle_roi[0],
                                                    end_rad=angle_roi[1])

            sweep_bool = Sample.angles_in_sweep(angles_array=diff_angles, start=min_, end=max_,
                                                sweep_type=range_, return_bool=True).reshape(-1)

        else:
            sweep_bool = True
        return sweep_bool

    def gen_bool_mask(self, run_data):
        d_bool = self.gen_d_roi_bool_mask(run_data['d_spacings'])
        sweep_bool = self.gen_angle_roi_bool_mask(angle_roi=run_data['angle_roi'], diff_angles=run_data['diff_angles'])
        bool_ = d_bool & sweep_bool
        return bool_

    def calc_cumulative_completeness(self, order_by_decrement=True, permutation_indices=None):
        completeness_list = []
        if not order_by_decrement and permutation_indices:
            runs_data = self.shuffle_dict_by_permutation(self.data_container, permutation_indices)
        else:
            runs_data = self.data_container
        ommited_list = []
        runs_order = []
        dynamic_ommition = []
        for run_data in runs_data.values():
            ommited_list.append(run_data['ommited'])

        hkl_original_base = np.array([])
        while not all(ommited_list):
            if order_by_decrement:
                completeness, index, hkl_add = self.calc_highest_comp_increment(hkl_original_base=hkl_original_base,
                                                                                runs_data=runs_data)
            else:
                completeness, index, hkl_add = self.calc_dict_order_comp(hkl_original_base=hkl_original_base,
                                                                         runs_data=runs_data)
            completeness_list.append(completeness)
            runs_data[index]['ommited'] = True
            dynamic_ommition.append(index)
            ommited_list[index] = True
            runs_order.append(index)
            hkl_original_base = np.append(hkl_original_base, hkl_add)
        for index in dynamic_ommition:
            runs_data[index]['ommited'] = False
        return completeness_list, runs_order

    def calc_highest_comp_increment(self, hkl_original_base, runs_data):
        completeness = None
        index = None
        hkl_original = None
        for i, run_data in runs_data.items():
            if not run_data['ommited']:
                bool_mask = self.gen_bool_mask(run_data)
                hkl_original_ = run_data['hkl_origin_e'][bool_mask].reshape(-1)
                completeness_ = self.calc_completeness(base_hkl=hkl_original_base, hkl=hkl_original_)
                if not completeness or completeness < completeness_:
                    completeness = completeness_
                    index = i
                    hkl_original = hkl_original_

        return completeness, index, hkl_original

    def calc_dict_order_comp(self, hkl_original_base, runs_data):
        for index, run_data in runs_data.items():
            if not run_data['ommited']:
                bool_mask = self.gen_bool_mask(run_data=run_data)
                hkl_original = run_data['hkl_origin_e'][bool_mask].reshape(-1)
                completeness = self.calc_completeness(base_hkl=hkl_original_base, hkl=hkl_original)
                return completeness, index, hkl_original

    def shuffle_dict_by_permutation(self, original_dict, permutation):
        items = list(original_dict.items())
        shuffled_items = [items[i] for i in permutation]
        return dict(shuffled_items)

    def calc_completeness(self, hkl, bool_mask=None, base_hkl=None):
        assert self.hkl_orig_all_e.any(), 'to sort by comp, first add hkl origin super group'
        if bool_mask is not None:
            hkl = hkl[bool_mask]
        if base_hkl is not None:
            hkl = np.append(hkl, base_hkl)
        completeness = self.completeness(hkl_original=hkl, all_original_hkl=self.hkl_orig_all_e, cut_d=True)
        return completeness

    def all_runs_bool_masks(self):
        bool_masks = []
        for run_data in self.data_container.values():
            bool_mask_ = self.gen_bool_mask(run_data=run_data)
            bool_masks.append(bool_mask_)
        all_origin_mask = self.gen_d_roi_bool_mask(self.d_spacings_all)
        return bool_masks, all_origin_mask

    def completeness(self, hkl_original, all_original_hkl, cut_d=False):
        if cut_d:
            bool_a_ = self.gen_d_roi_bool_mask(self.d_spacings_all)
            n_uniq_hkl = len(np.unique(hkl_original))
            n_uniq_hkl_all = len(np.unique(all_original_hkl[bool_a_]))
            completeness = n_uniq_hkl / n_uniq_hkl_all * 100
            return completeness

        n_uniq_hkl = len(np.unique(hkl_original))
        n_uniq_hkl_all = len(np.unique(all_original_hkl))
        completeness = n_uniq_hkl / n_uniq_hkl_all * 100
        return completeness


class ScanDataContainer:
    def __init__(self, diff_vecs, hkl, hkl_origin, diff_angles, scan_setup, start_angle, sweep):
        self.diff_vecs = diff_vecs
        self.hkl = hkl
        self.hkl_origin = hkl_origin
        self.diff_angles = diff_angles
        self.scan_setup = scan_setup
        self.start_angle = start_angle
        self.sweep = sweep


class StrategyContainer:
    scan_data_containers = []

    def add_scan_data_container(self, sdc):
        assert isinstance(sdc, ScanDataContainer), 'Wrong scan data format!'
        self.scan_data_containers.append(sdc)

    def clear_data(self):
        self.scan_data_containers = []

    def get_hkl(self):
        hkl = []
        for scan in self.scan_data_containers:
            hkl.append(scan.hkl)
        return hkl

    def get_hkl_roi(self, d_low, d_high, diffraction_angles_roi):
        pass

    def get_hkl_origin(self):
        hkl_origin = []
        for scan in self.scan_data_containers:
            hkl_origin.append(scan.hkl_origin)
        return hkl_origin

    def hasdata(self):
        if self.scan_data_containers:
            return True
        else:
            return False


def create_cumulative_fig(ordered_hkl_orig, run_indices, parameters, hkl_origin_in_d_range):
    comp_y, comp_x = [], []
    hkl_origin_ = np.array([]).reshape(-1, 3)
    for hkl_ in ordered_hkl_orig:
        hkl_origin_ = np.vstack((hkl_origin_, hkl_))
        comp_y_, comp_x_ = generate_completeness_plot(hkl_orig=hkl_origin_, all_hkl_orig=hkl_origin_in_d_range,
                                                      parameters=parameters, step=.1)
        comp_y.append(comp_y_)
        comp_x.append(comp_x_)
    fig = go.Figure()
    fig.add_hline(y=100, line_dash="dash", line_color="black", annotation_text="100",
                  annotation_position="bottom right")

    for n, (x, y) in enumerate(zip(comp_x, comp_y)):
        name = ' + '.join([str(i) for i in run_indices[:n + 1]])

        fig.add_trace((go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',  # Shows both lines and markers
            name=f'Runs {name}',
            fill='tonexty'
        )))
    fig.update_layout(
        title='Cumulative completeness',
    )
    fig.update_traces(line_width=3)
    fig.update_xaxes({'title': 'd, Å'})
    fig.update_yaxes({'title': '%'})
    fig['layout']['xaxis']['autorange'] = "reversed"
    return fig


def cumulative_dag_row_data_processing(dag_data):
    if not dag_data:
        return {
            'run n': [],
            'run_order': [],
            'min': [],
            'max': [],
            'min_angles': [],
            'max_angles': [],
            'selected': [],
        }

    output_data = {
        'run n': [],
        'run_order': [],
        'min': [],
        'max': [],
        'min_angles': [],
        'max_angles': [],
        'selected': [],
    }

    error_messages = []
    has_errors = False

    for run_data in dag_data:
        for key in output_data:
            if key not in ('selected', 'run n', 'run_order'):
                output_data[key].append(float(run_data[key]))
            else:
                output_data[key].append(run_data[key])

        if not run_data['selected']:
            continue

        run_n = run_data['run n']
        min_val = float(run_data['min'])
        max_val = float(run_data['max'])
        min_lim = float(run_data['min_angles'])
        max_lim = float(run_data['max_angles'])

        if min_val is None or max_val is None or min_lim is None or max_lim is None:
            continue

        run_error_messages = []
        if min_val < min_lim or min_val > max_lim:
            run_error_messages.append(f"min value {min_val} is out of range [{min_lim}, {max_lim}]")
        if max_val < min_lim or max_val > max_lim:
            run_error_messages.append(f"max value {max_val} is out of range [{min_lim}, {max_lim}]")

        if run_error_messages:
            has_errors = True
            error_messages.append(f"Run n {run_n}:\n" + "\n".join(run_error_messages))

    if has_errors:
        error_message = "Errors occurred:\n" + "\n".join(error_messages)
        modal_tuple_ = MODAL_TUPLE(header='Angles out of range', body=error_message)
        raise CalcCumulativeCompletenessError(modal_tuple_)

    return output_data


def generate_rowdata_cumulative_ag(exp_inst):
    sweeps = []
    start_angles = []
    end_angles = []
    run_n = []
    editable_flag = True
    for n, run in enumerate(exp_inst.strategy_data_container.scan_data_containers):
        sweep = run.sweep
        start_angle = run.start_angle
        sweeps.append(sweep)
        start_angles.append(start_angle)
        if (sweep is not None) & (start_angle is not None):
            end_angles.append(start_angle + sweep)
        else:
            end_angles.append(None)
            editable_flag = False
        run_n.append(n)
    if editable_flag:
        min_angles = [min(s, e) for s, e in zip(start_angles, end_angles)]
        max_angles = [max(s, e) for s, e in zip(start_angles, end_angles)]
    else:
        min_angles = [None] * len(sweeps)
        max_angles = [None] * len(sweeps)

    df = pd.DataFrame({
        'id': ['cumulative_data_row_' + str(i) for i in run_n],
        'run n': run_n,
        'min': min_angles,
        'max': max_angles,
        'selected': [True] * len(run_n),
        'completeness': [None] * len(run_n),
        'run_order': run_n,
        'min_angles': min_angles,
        'max_angles': max_angles,
    })

    df['tooltip'] = [
        f"Angle range: from {a:.2f} to {b:.2f}"
        for a, b in zip(df['min_angles'], df['max_angles'])
    ]
    return df.to_dict("records")


def process_d_range(exp_inst, d_range):
    low_workaround = 0.0
    high_workaround = 9999.0

    dmin = d_range[0]['d_min']
    dmax = d_range[0]['d_max']
    dmin = low_workaround if dmin in (None, '') else float(dmin)
    dmax = high_workaround if dmax in (None, '') else float(dmax)
    dmin, dmax = sorted([dmin, dmax])

    if (dmin == low_workaround) and (dmax == high_workaround):
        exp_inst.cdcc.set_d_roi(None)
    else:
        exp_inst.cdcc.set_d_roi((dmin, dmax))

    return dmin, dmax


@mylogger('DEBUG', log_args=True)
def update_data_container(exp_inst, input_parameters, dag_length):
    for i in range(dag_length):
        run_n = input_parameters['run n'][i]

        if not input_parameters['selected'][i]:
            exp_inst.cdcc.data_container[run_n]['ommited'] = True
            continue

        exp_inst.cdcc.data_container[run_n]['ommited'] = False

        min_val = input_parameters['min'][i]
        max_val = input_parameters['max'][i]
        min_angle = input_parameters['min_angles'][i]
        max_angle = input_parameters['max_angles'][i]

        if {min_val, max_val} == {min_angle, max_angle}:
            exp_inst.cdcc.data_container[run_n]['angle_roi'] = None
        else:
            angle_roi = [
                min(min_val, max_val),
                max(min_val, max_val)
            ]
            exp_inst.cdcc.data_container[run_n]['angle_roi'] = [
                np.deg2rad(angle) for angle in angle_roi
            ]


def update_completeness_data(dag_data, completeness):
    for n, run in enumerate(dag_data):
        run['completeness'] = completeness.get(n, None)
    return dag_data


def prepare_xy_grid_flatten(xy_ranges, xy_steps):
    x_range = np.arange(xy_ranges[0][0], xy_ranges[0][1], xy_steps[0])
    y_range = np.arange(xy_ranges[1][0], xy_ranges[1][1], xy_steps[1])
    x_range_new, y_range_new = np.meshgrid(x_range, y_range)
    return x_range_new.reshape(-1), y_range_new.reshape(-1)


def hkl_to_str(hkl_array):
    hkl_str = np.char.add(np.char.add(hkl_array[:, 0].astype(str), ' '),
                          np.char.add(hkl_array[:, 1].astype(str), ' '))
    hkl_str = np.char.add(hkl_str, hkl_array[:, 2].astype(str))

    return hkl_str


def generate_random_hex_colors(num_colors):
    hex_colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        hex_color = "#{:06x}".format(r << 16 | g << 8 | b)
        hex_colors.append(hex_color)
    return hex_colors


def find_index_digit(str_):
    pattern = r'[A-Za-z_]*(\d+).*'
    index = re.match(pattern, str_)[1]
    return index


def prepare_hidden_columns(table_index, data):
    hidden_columns = []
    if data['obst_prm_' + table_index + '_geometry'] == 'rectangle':
        hidden_columns.append('obst_prm_' + table_index + '_diameter')
    else:
        hidden_columns.append('obst_prm_' + table_index + '_height')
        hidden_columns.append('obst_prm_' + table_index + '_width')
    if data['obst_prm_' + table_index + '_orientation'] == 'normal':
        hidden_columns.append('obst_prm_' + table_index + '_displacement_y')
        hidden_columns.append('obst_prm_' + table_index + '_displacement_z')
    return hidden_columns


if __name__ == '__main__':
    load_hklf4('Z4.hkl')
