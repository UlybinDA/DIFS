from dash import dash_table, html
from dash.dash_table.Format import Format, Scheme
import pandas as pd
import dash_ag_grid as dag
from my_logger import mylogger


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
                'options': [{'label': f'{name}', 'value': no} for
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
            dropdown_options_list += [{'label': f'{names[no]}', 'value': f'{no}'}, ]
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


def generate_empty_dag_for_cumulative_completeness(id_):
    df = pd.DataFrame({
        'id': [None],
        'run n': [None],
        'min': [None],
        'max': [None],
        'selected': [None],
        'completeness': [None],
        'run_order': [None],
        'min_angles': [None],
        'max_angles': [None],
    })

    df['tooltip'] = ['fill me in']

    column_defs = [
        {"field": "id", "hide": True},
        {"field": "run n", "rowDrag": True},
        {
            "field": "min",
            "headerName": "Min Angle",
            "tooltipField": "tooltip",
            "editable": {"function": "params.data.min_angles !== null"},
            "cellClassRules": {
                "invalid-value": "params.value < params.data.min_angles || params.value > params.data.max_angles"
            },
            "valueSetter": {
                "function": """
                        function(params) {
                            const val = parseFloat(params.newValue);
                            const minANG = params.data.min_angles;
                            const maxANG = params.data.max_angles;
                            if (isNaN(val) || val < minANG || val > maxANG) {
                                return false;
                            }
                            params.data.min = val;
                            return true;
                        }
                    """
            }
        },
        {
            "field": "max",
            "headerName": "Max Angle",
            "tooltipField": "tooltip",
            "editable": {"function": "params.data.min_angles !== null"},
            "cellClassRules": {
                "invalid-value": "params.value < params.data.min_angles || params.value > params.data.max_angles"
            },
            "valueSetter": {
                "function": """
                        function(params) {
                            const val = parseFloat(params.newValue);
                            const minANG = params.data.min_angles;
                            const maxANG = params.data.max_angles;
                            if (isNaN(val) || val < minANG || val > maxANG) {
                                return false;
                            }
                            params.data.max = val;
                            return true;
                        }
                    """
            }
        },
        {
            "field": "selected",
            "headerName": "Select",
            "cellRenderer": "agCheckboxCellRenderer",
            "cellEditor": "agCheckboxCellEditor",
            "editable": True,
            "width": 100,
            "valueGetter": {"function": "params.data.selected || false"},
            "valueSetter": {
                "function": """
                        function(params) {
                            params.data.selected = params.newValue;
                            return true;
                        }
                    """
            }
        },
        {"field": "completeness", "headerName": "Completeness", "sortable": True},
        {"field": "run_order", "hide": True},
        {"field": "min_angles", "hide": True},
        {"field": "max_angles", "hide": True},
        {"field": "tooltip", "hide": True}
    ]

    grid_options = {
        "rowDragManaged": True,
        "animateRows": True,
        "suppressMoveWhenRowDragging": False,
        "immutableData": True,
        "suppressMovableColumns": True,
        "deltaRowDataMode": True,
        "getRowNodeId": {"function": "params.data.id"},
        "enableBrowserTooltips": True,
        "tooltipShowDelay": 0,
    }

    dag_table = dag.AgGrid(
        id="completeness_dag",
        columnDefs=column_defs,
        rowData=df.to_dict("records"),
        dashGridOptions=grid_options,
        defaultColDef={"sortable": False, "filter": False, "resizable": True},
        persistence=False,
        persistence_type='memory',
        style={"height": "500px", "width": "50%"}
    )

    return dag_table


def generate_dag_for_cumulative_completeness(exp_instance):
    sweeps = []
    start_angles = []
    end_angles = []
    run_n = []
    editable_flag = True
    for n, run in enumerate(exp_instance.strategy_data_container.scan_data_containers):
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

    column_defs = [
        {"field": "id", "hide": True},
        {"field": "run n", "rowDrag": True},
        {
            "field": "min",
            "headerName": "Min Angle",
            "tooltipField": "tooltip",
            "editable": {"function": "params.data.min_angles !== null"},
            "cellClassRules": {
                "invalid-value": "params.value < params.data.min_angles || params.value > params.data.max_angles"
            },
            "valueSetter": {
                "function": """
                    function(params) {
                        const val = parseFloat(params.newValue);
                        const minANG = params.data.min_angles;
                        const maxANG = params.data.max_angles;
                        if (isNaN(val) || val < minANG || val > maxANG) {
                            return false;
                        }
                        params.data.min = val;
                        return true;
                    }
                """
            }
        },
        {
            "field": "max",
            "headerName": "Max Angle",
            "tooltipField": "tooltip",
            "editable": {"function": "params.data.min_angles !== null"},
            "cellClassRules": {
                "invalid-value": "params.value < params.data.min_angles || params.value > params.data.max_angles"
            },
            "valueSetter": {
                "function": """
                    function(params) {
                        const val = parseFloat(params.newValue);
                        const minANG = params.data.min_angles;
                        const maxANG = params.data.max_angles;
                        if (isNaN(val) || val < minANG || val > maxANG) {
                            return false;
                        }
                        params.data.max = val;
                        return true;
                    }
                """
            }
        },
        {
            "field": "selected",
            "headerName": "Select",
            "cellRenderer": "agCheckboxCellRenderer",
            "cellEditor": "agCheckboxCellEditor",
            "editable": True,
            "width": 100,
            "valueGetter": {"function": "params.data.selected || false"},
            "valueSetter": {
                "function": """
                    function(params) {
                        params.data.selected = params.newValue;
                        return true;
                    }
                """
            }
        },
        {"field": "completeness", "headerName": "Completeness"},
        {"field": "run_order", "hide": True},
        {"field": "min_angles", "hide": True},
        {"field": "max_angles", "hide": True},
        {"field": "tooltip", "hide": True}
    ]

    grid_options = {
        "rowDragManaged": True,
        "animateRows": True,
        "suppressMoveWhenRowDragging": False,
        "immutableData": True,
        "suppressMovableColumns": True,
        "deltaRowDataMode": True,
        "getRowNodeId": {"function": "params.data.id"},
        "enableBrowserTooltips": True,
        "tooltipShowDelay": 0,
    }

    dag_table = dag.AgGrid(
        id="completeness_dag",
        columnDefs=column_defs,
        rowData=df.to_dict("records"),
        dashGridOptions=grid_options,
        defaultColDef={"sortable": True, "filter": False, "resizable": True},
        persistence=False,
        persistence_type='memory',
        style={"height": "500px", "width": "50%"}
    )

    return dag_table


def get_range_dag(id_):
    return html.Div(
        dag.AgGrid(
            id=id_,
            columnDefs=[
                {
                    "field": "d_min",
                    "headerName": "Min",
                    "editable": True,
                    "width": 70,
                    "valueParser": {
                        "function": """
                                function(params) {
                                    const value = parseFloat(params.newValue);
                                    return isNaN(value) ? null : value;
                                }
                            """
                    },
                    "cellEditor": "agTextCellEditor",
                },
                {
                    "field": "d_max",
                    "headerName": "Max",
                    "editable": True,
                    "width": 70,
                    "valueParser": {
                        "function": """
                                function(params) {
                                    const value = parseFloat(params.newValue);
                                    return isNaN(value) ? null : value;
                                }
                            """
                    },
                    "cellEditor": "agTextCellEditor",
                },
            ],
            rowData=[{"d_min": None, "d_max": None}],
            rowClassRules={
                "invalid-row": "(params.data.d_min !== null && params.data.d_max !== null && params.data.d_min > params.data.d_max && params.data.d_min !== undefined && params.data.d_max !== undefined) || params.data.d_min < 0 || params.data.d_max < 0"
            },
            style={
                "height": "90px",
                "width": "160px",
                "fontSize": "10px",
            },
        ),
        style={"display": "inline-block", "padding": "2px"},
    )

@mylogger('DEBUG',log_args=True)
def get_diff_map_detector(id_):
    df = pd.DataFrame({
        'factor_detector': [False],
        'd_dist': [40],
        'rot_x': [0],
        'rot_y': [0],
        'rot_z': [0],
        'orientation': ['normal'],
        'disp_y': [0],
        'disp_z': [0],
    })
    return html.Div(
        dag.AgGrid(
            style={'width':1000,'height':85},
            id=id_,
            dashGridOptions={
                "onGridReady": {
                    "function": """
            function(params) {
                window.myGridApi = params.api;
                window.myColumnApi = params.columnApi;
            }
            """
                },
            },
            columnDefs=[
                {
                    "field": "factor_detector",
                    "headerName": "account",
                    "editable": True,
                    "width": 100,
                    "cellRenderer": "agCheckboxCellRenderer",
                    "cellEditor": "agCheckboxCellEditor",
                },
                {
                    "field": "d_dist",
                    "headerName": "Distance",
                    "editable": True,
                    "width": 90,
                    "hide": False,
                },
                {
                    "field": "rot_x",
                    "headerName": "Rotation x",
                    "editable": True,
                    "width": 100,
                },
                {
                    "field": "rot_y",
                    "headerName": "Rotation y",
                    "editable": True,
                    "width": 100,
                },
                {
                    "field": "rot_z",
                    "headerName": "Rotation z",
                    "editable": True,
                    "width": 100,
                },
                {
                    "field": "orientation",
                    "headerName": "Orientation",
                    "editable": True,
                    "width": 100,
                    "cellEditor": "agSelectCellEditor",
                    "cellEditorParams": {"values": ["normal", "independent"]},

                },
                {
                    "field": "disp_y",
                    "headerName": "Disp y",
                    "editable": {
                        "function": "params.data.orientation === 'independent' && Object.prototype.toString.call(params.data.geometry) === '[object String]'"},
                    "width": 80,
                    "cellClassRules":{
                        "hide-cell": "params.data.orientation === 'normal'"
                    },
                    'headerClass':"hide-cell",
                },
                {
                    "field": "disp_z",
                    "headerName": "Disp z",
                    "editable": {
                        "function": "params.data.orientation === 'independent' && Object.prototype.toString.call(params.data.geometry) === '[object String]'"},
                    "width": 80,
                    "cellClassRules": {
                        "hide-cell": "params.data.orientation === 'normal'"
                    },
                    'headerClass':"hide-cell",

                },

            ],
            rowData=df.to_dict('records'),

        )
    )

