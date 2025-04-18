from dash import dash_table, html
from dash.dash_table.Format import Format, Scheme
import pandas as pd


def generate_choose_scan_dropdown(exp_inst, id_, style, style_cell):
    if exp_inst.axes_rotations is None:
        return list()
    real_axes = exp_inst.axes_real
    input_axes_table = html.Div(dash_table.DataTable(
        data=pd.DataFrame({'x_axis': '', }, index=[0]).to_dict('records'),
        columns=[
            {'id': 'x_axis', 'name': 'Scan', 'type': 'numeric', 'presentation': 'dropdown'},
        ],
        editable=True,
        style_cell=style_cell,
        fill_width=False,
        dropdown={
            'x_axis': {
                'options': [{'label': f'{name}', 'value': no } for
                            no, name in enumerate(exp_inst.axes_names) if real_axes[no] == 'true']}, },
        id=f'{id_}'
    ),
        style=style
    )
    return input_axes_table


def generate_angle_table(exp_inst, id_, style, style_cell):
    if exp_inst.axes_rotations is None:
        return list()
    real_axes = exp_inst.axes_real
    input_angles_table = html.Div(dash_table.DataTable(
        data=pd.DataFrame(
            {f'{name}_{no}': exp_inst.axes_angles[no] for no, name in enumerate(exp_inst.axes_names)},
            index=[0]).to_dict('records'),
        columns=[{'id': f'{name}_{no}', 'name': f'{name}', 'type': 'numeric'} if real_axes[no] == 'true' else {
            'id': f'{name}_{no}', 'name': f'{name}', 'type': 'numeric', 'editable': False} for no, name in
                 enumerate(exp_inst.axes_names)
                 ],
        editable=True,
        style_cell=style_cell,
        fill_width=False,
        id=id_,
    ),
        style=style
    )
    return input_angles_table


def generate_obst_table(n_cl, data=None):
    if not data:
        new_div_table = html.Div([
            html.Div([dash_table.DataTable(fill_width=False,
                                           id={'type': 'obstacle_table', 'index': n_cl},
                                           editable=True,
                                           style_cell={
                                               'width': '100px'
                                           },
                                           data=pd.DataFrame(
                                               {f'obst_prm_{n_cl}_distance': 10,
                                                f'obst_prm_{n_cl}_geometry': '',
                                                f'obst_prm_{n_cl}_orientation': 'normal',
                                                f'obst_prm_{n_cl}_rotation_x': 0,
                                                f'obst_prm_{n_cl}_rotation_y': 0,
                                                f'obst_prm_{n_cl}_rotation_z': 0,
                                                f'obst_prm_{n_cl}_height': '',
                                                f'obst_prm_{n_cl}_width': '',
                                                f'obst_prm_{n_cl}_diameter': '',
                                                f'obst_prm_{n_cl}_displacement_y': '',
                                                f'obst_prm_{n_cl}_displacement_z': '',
                                                f'obst_prm_{n_cl}_name': ''
                                                }, index=[0])
                                           .to_dict('records'),
                                           columns=[
                                               {'id': f'obst_prm_{n_cl}_distance', 'name': 'distance',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_geometry', 'name': 'geometry', 'type': 'text',
                                                'presentation': 'dropdown'},
                                               {'id': f'obst_prm_{n_cl}_orientation', 'name': 'orientation',
                                                'type': 'text', 'presentation': 'dropdown'},
                                               {'id': f'obst_prm_{n_cl}_rotation_x', 'name': 'rotation_x',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_rotation_y', 'name': 'rotation_y',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_rotation_z', 'name': 'rotation_z',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_height', 'name': 'height',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_width', 'name': 'width',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_diameter', 'name': 'diameter',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_displacement_y', 'name': 'displacement y',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_displacement_z', 'name': 'displacement z',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_name', 'name': f'name',
                                                'type': 'text'}
                                           ],
                                           dropdown={
                                               f'obst_prm_{n_cl}_geometry': {
                                                   'options': [{'label': 'Circle', 'value': 'circle'},
                                                               {'label': 'Rectangle', 'value': 'rectangle'}]},
                                               f'obst_prm_{n_cl}_orientation': {
                                                   'options': [{'label': 'Normal', 'value': 'normal'},
                                                               {'label': 'Independent', 'value': 'independent'}]},

                                           },
                                           )],
                     style={'display': 'inline-block'}
                     ),
            html.Div([html.Button('x', id={'type': 'obstacle_delete_div_button', 'index': n_cl}
                                  , n_clicks=0
                                  ), ],
                     style={'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-left': '0vw',
                            'margin-top': '1vw'})
        ],
            id={'type': 'obstacle_div_table'}
        )
    else:
        new_div_table = html.Div([
            html.Div([dash_table.DataTable(fill_width=False,
                                           id={'type': 'obstacle_table', 'index': n_cl},
                                           editable=True,
                                           style_cell={
                                               'width': '100px'
                                           },
                                           data=pd.DataFrame(
                                               {f'obst_prm_{n_cl}_distance': data['distance'],
                                                f'obst_prm_{n_cl}_geometry': data['geometry'],
                                                f'obst_prm_{n_cl}_orientation': data['orientation'],
                                                f'obst_prm_{n_cl}_rotation_x': data['rotation_x'],
                                                f'obst_prm_{n_cl}_rotation_y': data['rotation_y'],
                                                f'obst_prm_{n_cl}_rotation_z': data['rotation_z'],
                                                f'obst_prm_{n_cl}_height': data['height'],
                                                f'obst_prm_{n_cl}_width': data['width'],
                                                f'obst_prm_{n_cl}_diameter': data['diameter'],
                                                f'obst_prm_{n_cl}_displacement_y': data['displacement_y'],
                                                f'obst_prm_{n_cl}_displacement_z': data['displacement_z'],
                                                f'obst_prm_{n_cl}_name': data['name']
                                                }, index=[0])
                                           .to_dict('records'),
                                           columns=[
                                               {'id': f'obst_prm_{n_cl}_distance', 'name': 'distance',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_geometry', 'name': 'geometry', 'type': 'text',
                                                'presentation': 'dropdown'},
                                               {'id': f'obst_prm_{n_cl}_orientation', 'name': 'orientation',
                                                'type': 'text', 'presentation': 'dropdown'},
                                               {'id': f'obst_prm_{n_cl}_rotation_x', 'name': 'rotation_x',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_rotation_y', 'name': 'rotation_y',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_rotation_z', 'name': 'rotation_z',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_height', 'name': 'height',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_width', 'name': 'width',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_diameter', 'name': 'diameter',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_displacement_y', 'name': 'displacement y',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_displacement_z', 'name': 'displacement z',
                                                'type': 'numeric', 'format': Format(precision=6)},
                                               {'id': f'obst_prm_{n_cl}_name', 'name': f'name',
                                                'type': 'text'}
                                           ],
                                           dropdown={
                                               f'obst_prm_{n_cl}_geometry': {
                                                   'options': [{'label': 'Circle', 'value': 'circle'},
                                                               {'label': 'Rectangle', 'value': 'rectangle'}]},
                                               f'obst_prm_{n_cl}_orientation': {
                                                   'options': [{'label': 'Normal', 'value': 'normal'},
                                                               {'label': 'Independent', 'value': 'independent'}]},

                                           },
                                           )],
                     style={'display': 'inline-block'}
                     ),
            html.Div([html.Button('x', id={'type': 'obstacle_delete_div_button', 'index': n_cl}
                                  , n_clicks=0
                                  ), ],
                     style={'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-left': '0vw',
                            'margin-top': '1vw'})
        ],
            id={'type': 'obstacle_div_table'}
        )
    return new_div_table


def gen_run_table(real_axes, axes_angles, rotations, names, table_num, data=None):
    dropdown_options_list = list()

    for no, rot in enumerate(rotations):
        if real_axes[no] == 'true':
            dropdown_options_list += [{'label': f'{names[no]}', 'value': f'{no }'}, ]
    if data is None:
        dict_for_table = {
            f'{table_num}scan_no': '',
            f'{table_num}det_dist': 40,
            f'{table_num}det_orientation': 'normal',
            f'{table_num}det_rot_x': 0,
            f'{table_num}det_rot_y': 0,
            f'{table_num}det_rot_z': 0,
            f'{table_num}det_disp_y': '',
            f'{table_num}det_disp_z': '',
            f'{table_num}scan_sweep': 180,
        }

    columns_list1 = [
        {'id': f'{table_num}scan_no', 'name': 'No of scan', 'type': 'text', 'presentation': 'dropdown'},
        {'id': f'{table_num}scan_sweep', 'name': 'sweep', 'type': 'numeric'}
    ]
    columns_list2 = [
        {'id': f'{table_num}det_dist', 'name': 'Det dist', 'type': 'numeric', 'format': Format(precision=2)},
        {'id': f'{table_num}det_orientation', 'name': 'Det orientation', 'type': 'text', 'presentation': 'dropdown'},
        {'id': f'{table_num}det_rot_x', 'name': 'Det rot x', 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.decimal_integer)},
        {'id': f'{table_num}det_rot_y', 'name': 'Det rot y', 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.decimal_integer)},
        {'id': f'{table_num}det_rot_z', 'name': 'Det rot z', 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.decimal_integer)},
        {'id': f'{table_num}det_disp_y', 'name': 'Det displace y', 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.decimal_integer)},
        {'id': f'{table_num}det_disp_z', 'name': 'Det displace z', 'type': 'numeric',
         'format': Format(precision=2, scheme=Scheme.decimal_integer)},
    ]

    if data is None:
        for num, rot in enumerate(rotations):
            dict_for_table[f'{table_num}_{num}_rot'] = 0
            dict_for_table.update({f'{table_num}_{num}_rot': axes_angles[num]})
            if real_axes[num] == 'true':
                columns_list1.append(
                    {'id': f'{table_num}_{num}_rot', 'name': f'{names[num]}', 'type': 'numeric',
                     'format': Format(precision=2, scheme=Scheme.decimal_integer)})

            elif real_axes[num] == 'false':
                columns_list1.append(
                    {'id': f'{table_num}_{num}_rot', 'name': f'{names[num]}', 'type': 'numeric',
                     'format': Format(precision=2, scheme=Scheme.decimal_integer), 'editable': False})
    else:
        dict_for_table = data
        for num, rot in enumerate(rotations):
            if real_axes[num] == 'true':
                columns_list1.append(
                    {'id': f'{table_num}_{num}_rot', 'name': f'{names[num]}', 'type': 'numeric',
                     'format': Format(precision=2, scheme=Scheme.decimal_integer)})

            elif real_axes[num] == 'false':
                columns_list1.append(
                    {'id': f'{table_num}_{num}_rot', 'name': f'{names[num]}', 'type': 'numeric',
                     'format': Format(precision=2, scheme=Scheme.decimal_integer), 'editable': False})

    columns_list1 = columns_list1 + columns_list2

    new_div_table = html.Div([
        html.Div([dash_table.DataTable(fill_width=False,
                                       id={'type': 'run_table', 'index': f'{table_num}'},
                                       editable=True,
                                       style_cell={
                                           'width': '50px'
                                       },
                                       data=pd.DataFrame(dict_for_table, index=[0]).to_dict('records'),
                                       columns=columns_list1,
                                       dropdown={
                                           f'{table_num}scan_no': {
                                               'options': dropdown_options_list
                                           },
                                           f'{table_num}det_orientation': {
                                               'options': [{'label': 'Normal', 'value': 'normal'},
                                                           {'label': 'Independent', 'value': 'independent'}]},
                                       },

                                       )],
                 style={'display': 'inline-block'}
                 ),
        html.Div([html.Button('x', id={'type': 'run_delete_div_button', 'index': table_num}
                              , n_clicks=0
                              ), ],
                 style={'display': 'inline-block',
                        'vertical-align': 'top',
                        'margin-left': '0vw',
                        'margin-top': '1vw'})
    ],
        id={'type': 'runs_div', 'index': table_num}
    )
    return new_div_table
