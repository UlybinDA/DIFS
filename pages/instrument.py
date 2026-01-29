from json import JSONDecodeError
from services.exceptions.exceptions import *
from dash import html, dcc, dash_table, Input, Output, State, ALL, MATCH, no_update as no_upd, ctx
from dash.dash_table.Format import Format, Scheme
import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
from os import listdir
import json
import os


from app import app
from global_state import content_vars, exp1
import services.service_functions as sf
from assets.modals_content import *
import assets.app_gens as apg
from logger.my_logger import mylogger




goniometer_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instruments', 'goniometers')
if not os.path.exists(goniometer_models_path):

    goniometer_models_path = os.path.join(os.path.dirname(__file__), 'instruments', 'goniometers')

goniometers = [i[:-5] for i in listdir(goniometer_models_path) if i.endswith('.json')]


children_div_complex_prm = [
    html.H6('Complex detector parameters'),
    dash_table.DataTable(fill_width=False,
                         id='complex_parameter_table',
                         data=content_vars.complex_detector_parameters,
                         editable=True,
                         columns=[{'name': i[:-4], 'id': i, 'type': 'numeric'} for i in
                                  list(pd.DataFrame.from_records(
                                      content_vars.complex_detector_parameters))],
                         style_cell={
                             'width': '30px'
                         }, )
]

modal2 = html.Div([
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(children='Header', id='modal_header2')),
        dbc.ModalBody(children='Body', id='modal_body2',
                      style={
                          'whiteSpace': 'pre-wrap',
                          'tabSize': '4',
                          'fontFamily': 'monospace'
                      }
                      )
    ],
        id='modal2',
        size='lg',
        is_open=False
    )
]
)

layout = html.Div([
    html.Div(list(), id='hidden_div_2', style={'display': 'none'}),
    modal2,
    html.H2('Instrument model'),
    html.Div([
        html.Button('Export instrument model', id='download_instrument_btn', n_clicks=0),
        dcc.Download(id='download_instrument'),
        dcc.Upload(html.Button('Import instrument model'),
                   accept='.json', multiple=False, max_size=100000, id='upload_instrument'),
        dcc.Dropdown(id='select_instrument', style={
            'vertical-align': 'top',
            'display': 'inline-block',
            'width': '500px',
        }), ]

    ),
    html.Div([html.H4('Wavelength, Å'),
              dcc.Input(
                  id='wavelength_input',
                  value=0.710730,
                  type='number',
                  min=0.1,
                  max=3,
                  step='any'

              ),
              html.Button(
                  'Set wavelength',
                  id='set_wavelength_button',
                  n_clicks=0
              ),
              dcc.Upload(html.Button('Import wavelength'), id='upload_wavelength',
                         accept='.json',
                         multiple=False, max_size=100),
              html.Button('Export wavelength', id='download_wavelength_btn', n_clicks=0),
              dcc.Download(id='download_wavelength')

              ]
             ),
    html.Div([
        html.H4('Goniometer emulation'),
        dash_table.DataTable(
            fill_width=False,
            id='Goniometer_table',
            data=content_vars.goniometer_table,
            editable=True,
            columns=[
                {'name': 'rotation', 'id': 'rotation_col', 'type': 'text', 'presentation': 'dropdown'},
                {'name': 'direction', 'id': 'direction_col', 'type': 'text', 'presentation': 'dropdown'},
                {'name': 'name', 'id': 'name_col', 'type': 'text', },
                {'name': 'real', 'id': 'real_col', 'type': 'text', 'presentation': 'dropdown'},
                {'name': 'angle, °', 'id': 'ang_col', 'type': 'numeric',
                 'format': Format(precision=1, scheme=Scheme.decimal_integer)},
            ],
            dropdown={
                'rotation_col': {
                    'options': [
                        {'label': 'x-rot', 'value': 'x'},
                        {'label': 'y-rot', 'value': 'y'},
                        {'label': 'z-rot', 'value': 'z'},
                    ]
                },
                'direction_col': {
                    'options': [
                        {'label': 'positive', 'value': '+'},
                        {'label': 'negative', 'value': '-'},
                    ]
                },
                'real_col': {
                    'options': [
                        {'label': 'true', 'value': True},
                        {'label': 'false', 'value': False},
                    ]
                },
            },

            style_cell={
                'width': '30px'
            }
        ),
        html.Button('Set goniometer',
                    id='Set_goniometer',
                    n_clicks=0,
                    style={
                        'vertical-align': 'top',
                        'height': '30px',
                    }
                    ),

        html.Button('+',
                    id='Goniometer_table_plus',
                    n_clicks=0,
                    style={
                        'height': '30px',
                        'width': '30px',
                    }
                    ),
        html.Button('-',
                    id='Goniometer_table_minus',
                    n_clicks=0,
                    style={
                        'height': '30px',
                        'width': '30px',
                    }
                    ),

    ]),
    html.Div((
        html.Button('Export goniometer input',
                    id='download_goniometer_btn',
                    n_clicks=0,
                    style={
                        'vertical-align': 'top',
                        'display': 'inline-block',
                        'height': '30px',
                    }
                    ),
        dcc.Download(id='download_goniometer', ),
        dcc.Upload(html.Button('Import goniometer', style={'display': 'inline-block'}), id='upload_goniometer',
                   accept='.json',
                   multiple=False, max_size=10000),
        dcc.Dropdown(options=goniometers, value='', id='goniometer_dropdown', clearable=False),),

        style={'width': '1000px'}),
    html.Div((
        dcc.Upload(
            html.Button('Import logic collision',
                        style={'display': 'inline-block'}),
            id='upload_collision_log',
            accept='.json',
            multiple=False, max_size=10000),
        dcc.Checklist(['factor collision'], id='collision_check'),

    )),
    html.Div([
        html.Div([
            html.H4('Detector model'),
            dcc.Dropdown(['Rectangle', 'Circle'], '', id='Det_geom_dropdown'),
        ],
            style={
                'display': 'inline-block',
                'width': '130px'
            }
        ),

        html.Div([
            dash_table.DataTable(fill_width=False,
                                 id='rectangle_parameter_table',
                                 data=content_vars.detector_geometry_rectangle_parameters,
                                 editable=True,
                                 columns=[{'name': i[:-4], 'id': i, 'type': 'numeric'} for i in
                                          list(pd.DataFrame.from_records(
                                              content_vars.detector_geometry_rectangle_parameters))],
                                 style_cell={
                                     'width': '30px'
                                 }, )
        ],
            id='div_rectangle_prm',
            style={'display': 'block'},
        ),
        html.Div([
            html.H6('Is complex?'),
            dcc.Checklist(['Complex'], id='Complex_check'),
        ],
            id='div_complex_check',
            style={
                'display': 'inline-block',
                'width': '130px'
            }
        ),
        html.Div('',
                 id='div_complex_prm',
                 style={
                     'display': 'inline-block',
                 }
                 ),
        html.Div([
            dash_table.DataTable(fill_width=False,
                                 id='circle_parameter_table',
                                 data=content_vars.detector_geometry_circle_parameters,
                                 editable=True,
                                 columns=[{'name': i[:-4], 'id': i, 'type': 'numeric'} for i in
                                          list(pd.DataFrame.from_records(
                                              content_vars.detector_geometry_circle_parameters))],
                                 style_cell={
                                     'width': '30px'
                                 }, )
        ],
            id='div_circle_prm',
            style={'display': 'block', },
        ),
        html.Div(
            html.Button(
                'Set Detector',
                n_clicks=0,
                id='Set_detector_button',
                style={'background-color': 'white'}
            )

        ),
        html.Button('Export detector input',
                    id='download_detector_btn',
                    n_clicks=0,
                    style={
                        'vertical-align': 'top',
                        'display': 'inline-block',
                        'height': '30px',
                    }
                    ),
        dcc.Download(id='download_detector', ),
        dcc.Upload(html.Button('Import detector', style={'display': 'inline-block'}), id='upload_detector',
                   accept='.json',
                   multiple=False, max_size=400),

    ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top', }
    ),
    html.Div([
        html.H4('Diamond anvil'),
        html.Div([html.H6(
            'Anvil_aperture'
        ),
            dcc.Input(
                id='anvil_aperture_input',
                value=35,
                type='number',
                min=0.1,
                max=89.9,
                step='any',

            ),
        ],
            style={
                'display': 'inline-block',
                'vertical-align': 'top', }
        ),
        html.Div([
            html.H6('Normal vector'),
            dash_table.DataTable(fill_width=False,
                                 id='anvil_normal_vector_table',
                                 data=pd.DataFrame(np.array([[1, 0, 0]]), columns=(
                                     'anvil_normal_x', 'anvil_normal_y', 'anvil_normal_z')).to_dict('records'),
                                 editable=True,
                                 columns=[
                                     {'name': 'x', 'id': 'anvil_normal_x', 'type': 'numeric'},
                                     {'name': 'y', 'id': 'anvil_normal_y', 'type': 'numeric'},
                                     {'name': 'z', 'id': 'anvil_normal_z', 'type': 'numeric'},
                                 ],
                                 style_cell={
                                     'width': '30px'
                                 }, )]
        ),
        html.Button('Set anvil', id='set_anvil_btn', n_clicks=0),
        dcc.Checklist(['factor anvil'], id='calc_anvil_check'),
    ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top', }
    ),
    html.Div([
        html.H4('Beam obstacles'),
        html.Button(
            'Add obstacle',
            id='add_obstacle_button',
            n_clicks=0
        ),
        html.Button(
            'Delete all obstacles',
            id='clear_obstacles_button',
            n_clicks=0
        ),
        html.Button(
            'Set obstacles',
            id='set_obstacles_button',
            n_clicks=0
        ),
        html.Button('Export obstacles input',
                    id='download_obstacles_btn',
                    n_clicks=0,
                    style={
                        'vertical-align': 'top',
                        'display': 'inline-block',
                        'height': '30px',
                    }
                    ),
        dcc.Download(id='download_obstacles', ),
        dcc.Upload(html.Button('Import obstacles', style={'display': 'inline-block'}), id='upload_obstacles',
                   accept='.json', multiple=False, max_size=10000),

    ], ),
    html.Div(list(),
             id='obstacle_div'
             ),
    html.Div([html.H4('Goniometer linked beam obstacles'),
              html.Button(
                  'Add obstacle',
                  id='add_linked_obstacle_button',
                  n_clicks=0
              ),
              html.Button(
                  'Delete all obstacles',
                  id='clear_linked_obstacles_button',
                  n_clicks=0
              ),
              html.Button(
                  'Set obstacles',
                  id='set_linked_obstacles_button',
                  n_clicks=0
              ),
              html.Button('Export obstacles input',
                          id='download_linked_obstacles_btn',
                          n_clicks=0,
                          style={
                              'vertical-align': 'top',
                              'display': 'inline-block',
                              'height': '30px',
                          }
                          ),
              dcc.Download(id='download_linked_obstacles', ),
              dcc.Upload(html.Button('Import obstacles', style={'display': 'inline-block'}),
                         id='upload_linked_obstacles',
                         accept='.json', multiple=False, max_size=10000),

              ]),
    html.Div(list(),
             id='linked_obstacle_div'
             )

]
)



def if_val_None_return_no_upd_else_return(val):
    if val is None:
        return no_upd
    else:
        return val

@app.callback(
    Output('wavelength_input', 'value', allow_duplicate=True),
    Output('Goniometer_table', 'data'),
    Output('goniometer_dropdown', 'value'),
    Output('Det_geom_dropdown', 'value'),
    Output('Complex_check', 'value'),
    Output('circle_parameter_table', 'data', allow_duplicate=True),
    Output('rectangle_parameter_table', 'data', allow_duplicate=True),
    Output('obstacle_div', 'children', allow_duplicate=True),
    Output('collision_check', 'value'),
    Output('anvil_normal_vector_table', 'data'),
    Output('anvil_aperture_input', 'value'),
    Output('calc_anvil_check', 'value'),
    Output('linked_obstacle_div', 'children', allow_duplicate=True),
    Output("page-1_stored_flag", "data", allow_duplicate=True),
    Input("page-1_stored_flag", "data"),
    State('stored_wavelength_val', 'data'),
    State('stored_goniometer_table', 'data'),
    State('stored_goniometer_dropdown', 'data'),
    State('stored_detector_dropdown', 'data'),
    State('stored_detector_check_complex', 'data'),
    State('stored_circle_parameters', 'data'),
    State('stored_rectangle_parameters', 'data'),
    State('stored_obstacles_div', 'data'),
    State('stored_log_collision_check', 'data'),
    State('stored_anvil_normal_data', 'data'),
    State('stored_anvil_aperture_data', 'data'),
    State('stored_anvil_calc_check', 'data'),
    State('stored_linked_obstacles_div', 'data'),
    prevent_initial_call=True)
@mylogger(level='DEBUG')
def get_stored_page_1_data(flag, wavelength_val, goniometer_table, goniometer_dropdown, detector_dropdown,
                           detector_check_complex, circle_parameters, rectangle_parameters, obstacle_div,
                           log_col_check, anvil_normal, anvil_aperture, anvil_check, linked_obstacle_div):
    if not flag:
        raise dash.exceptions.PreventUpdate()
    print(linked_obstacle_div)
    storage_data_list = [wavelength_val,
                         goniometer_table,
                         goniometer_dropdown,
                         detector_dropdown,
                         detector_check_complex,
                         circle_parameters,
                         rectangle_parameters,
                         obstacle_div,
                         log_col_check,
                         anvil_normal,
                         anvil_aperture,
                         anvil_check,
                         linked_obstacle_div,
                         ]
    output_list = [if_val_None_return_no_upd_else_return(val) for val in storage_data_list]
    output_list += [False, ]
    return output_list


@app.callback(
    Output('Goniometer_table', 'data', allow_duplicate=True),
    Input('Goniometer_table_plus', 'n_clicks'),
    State('Goniometer_table', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def expand_goniometer(n_clicks, data):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    df_data = pd.DataFrame.from_records(data)

    new_row = pd.DataFrame({'rotation': '', 'direction': '', 'name': '', 'real': '', 'angle': 0}, index=[0])
    df_data = df_data._append(new_row)
    content_vars.goniometer_table = df_data.to_dict('records')
    return content_vars.goniometer_table


@app.callback(
    Output('Goniometer_table', 'data', allow_duplicate=True),
    Input('Goniometer_table_minus', 'n_clicks'),
    State('Goniometer_table', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def shorten_goniometer(n_clicks, data):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    df_data = pd.DataFrame.from_records(data)
    if len(df_data) == 1:
        raise dash.exceptions.PreventUpdate()
    df_data = df_data.iloc[:-1]
    content_vars.goniometer_table = df_data.to_dict('records')
    return content_vars.goniometer_table


@app.callback(Output('stored_runs_div', 'data', allow_duplicate=True),
          Output('hidden_div_2', 'children', allow_duplicate=True),
          Output('Goniometer_table', 'data', allow_duplicate=True),
          Output('stored_goniometer_table', 'data', allow_duplicate=True),
          Output('stored_goniometer_dropdown', 'data', allow_duplicate=True),
          Input('goniometer_dropdown', 'value'),
          prevent_initial_call=True
          )
@mylogger(level='DEBUG', log_args=True)
def select_goniometer(name):
    if not name:
        raise dash.exceptions.PreventUpdate()
    path = ''.join((goniometer_models_path, name, '.json'))
    with open(path, 'r') as jf:
        js = jf.read()

    try:
        data_list = json.loads(js)
    except JSONDecodeError:
        return no_upd, (True, read_json_error.header, read_json_error.body), no_upd, no_upd
    data, real, angles = exp1.load_instrument_unit(data_list, 'goniometer')
    if data:
        content_vars.axes_angles = angles
        content_vars.real_axes = real
        return None, no_upd, data, data, name
    else:
        return no_upd, (True, load_goniometer_error.header, load_goniometer_error.body), no_upd, no_upd


@app.callback(
    Output('Set_goniometer', 'style'),
    Output('stored_goniometer_table', 'data', allow_duplicate=True),
    Output('stored_runs_div', 'data', allow_duplicate=True),
    Input('Set_goniometer', 'n_clicks'),
    State('Goniometer_table', 'data'),
    prevent_initial_call=True

)
@mylogger(level='DEBUG', log_args=True)
def set_goniometer(n_clicks, data):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    array_data = np.array(pd.DataFrame.from_records(data))
    if array_data.shape[1] != 10:
        return {'background-color': 'red'}, no_upd, no_upd
    array_data = array_data[:, 5:]
    pd_data = pd.DataFrame.from_records(data)
    if None in array_data or '' in array_data or np.nan in array_data:
        return {'background-color': 'red'}, no_upd, no_upd
    rotations = ''.join(np.array(pd_data['rotation_col']))
    directions = ''.join(np.array(pd_data['direction_col']))
    name = list(np.array(pd_data['name_col']))
    real = list(np.array(pd_data['real_col']))
    print(real)
    angles = list(np.array(pd_data['ang_col']))
    content_vars.real_axes = real
    content_vars.axes_angles = angles
    directions = [(1 if i == '+' else -1) for i in directions]
    exp1.set_goniometer(axes_rotations=rotations, axes_directions=directions, axes_names=name, axes_angles=angles,
                        axes_real=real)
    exp1.delete_scan(all_=True)
    return {'background-color': 'green'}, data, None


@app.callback(
    Output('div_rectangle_prm', 'style'),
    Output('div_complex_check', 'style'),
    Output('div_circle_prm', 'style'),
    Input('Det_geom_dropdown', 'value'),
)
@mylogger(level='DEBUG')
def show_detector_parameters(value):
    if value == 'Rectangle':
        return {'display': 'block'}, {'display': 'inline-block'}, {'display': 'none'}
    elif value == 'Circle':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    else:
        return {'display': 'block'}, {'display': 'none'}, {'display': 'block'}


@app.callback(
    Output('div_complex_prm', 'children'),
    Output('stored_detector_check_complex', 'data'),
    Input('Complex_check', 'value'),
    State('Det_geom_dropdown', 'value')
)
@mylogger(level='DEBUG')
def show_complex_parameters(check, geom):
    if check == list() or check == None:
        return '', no_upd
    elif check[0] == 'Complex' and geom == 'Rectangle':
        return children_div_complex_prm, check
    else:
        return '', no_upd


@app.callback(
    Output('Set_detector_button', 'style'),
    Output('stored_circle_parameters', 'data'),
    Output('stored_rectangle_parameters', 'data'),
    Output('stored_detector_dropdown', 'data'),
    Input('Set_detector_button', 'n_clicks'),
    State('Det_geom_dropdown', 'value'),
    State('rectangle_parameter_table', 'data'),
    State('circle_parameter_table', 'data'),
    State('div_complex_prm', 'children'),
    State('Complex_check', 'value'),
)
@mylogger(level='DEBUG', log_args=True)
def set_detector(n_clicks, geometry, rectangle_prm_table_data, circle_prm_table_data, complex_prm, complex_check):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()

    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    if geometry == '':
        return {'background-color': 'red'}, no_upd, no_upd, no_upd
    if geometry == 'Circle':
        circle_prm_data = np.array(pd.DataFrame.from_records(circle_prm_table_data).iloc[0])[0]
        if circle_prm_data == '':
            return {'background-color': 'red'}, no_upd, no_upd, no_upd
        elif circle_prm_data <= 0:
            return {'background-color': 'red'}, no_upd, no_upd, no_upd
        else:
            exp1.set_detector_param(det_geometry='circle', det_diameter=circle_prm_data)
            return {'background-color': 'green'}, circle_prm_table_data, None, geometry
    elif geometry == 'Rectangle':
        rectangle_prm_data = np.array(pd.DataFrame.from_records(rectangle_prm_table_data).iloc[0])
        mask = rectangle_prm_data != 0
        mask1 = rectangle_prm_data.astype('str') != ''
        if np.count_nonzero(mask) != 2:
            return {'background-color': 'red'}, no_upd, no_upd
        if np.count_nonzero(mask1) != 2:
            return {'background-color': 'red'}, no_upd, no_upd
        mask = rectangle_prm_data > 0
        if np.count_nonzero(mask) != 2:
            return {'background-color': 'red'}
        if complex_check is not None and complex_check != list():

            if complex_prm == '':
                return {'background-color': 'red'}
            complex_data = np.array(pd.DataFrame.from_records(complex_prm[1]['props']['data']).iloc[0])
            mask = complex_data != 0
            mask1 = complex_data != ''
            if np.count_nonzero(mask) != 4:
                return {'background-color': 'red'}
            if np.count_nonzero(mask1) != 4:
                return {'background-color': 'red'}
            mask = complex_data > 0
            if np.count_nonzero(mask) != 4:
                return {'background-color': 'red'}
            try:
                exp1.set_detector_param(det_geometry='rectangle', det_height=rectangle_prm_data[0],
                                        det_width=rectangle_prm_data[1], det_complex=True,
                                        det_complex_format=(complex_data[0], complex_data[1]),
                                        det_row_col_spacing=(complex_data[2], complex_data[3]))
                return {'background-color': 'green'}, None, rectangle_prm_table_data, geometry
            except IndexError:
                return {'background-color': 'red'}

        else:
            try:
                exp1.set_detector_param(det_geometry='rectangle', det_height=rectangle_prm_data[0],
                                        det_width=rectangle_prm_data[1])
                return {'background-color': 'green'}, None, rectangle_prm_table_data, geometry
            except IndexError:
                return {'background-color': 'red'}, no_upd, no_upd


@app.callback(
    Output('download_detector', 'data'),
    Input('download_detector_btn', 'n_clicks'),
)
@mylogger(level='DEBUG')
def download_detector(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    json_data = exp1.json_export('detector')
    return dict(content=json_data, filename='detector_input.json')


@app.callback(
    Output('hidden_div_2', 'children', allow_duplicate=True),
    Output('upload_detector', 'contents'),  # workaround to upload from same dir multiple times
    Output('circle_parameter_table', 'data', allow_duplicate=True),
    Output('rectangle_parameter_table', 'data', allow_duplicate=True),
    Output('Det_geom_dropdown', 'value', allow_duplicate=True),
    Output('stored_circle_parameters', 'data', allow_duplicate=True),
    Output('stored_rectangle_parameters', 'data', allow_duplicate=True),
    Output('stored_detector_dropdown', 'data', allow_duplicate=True),
    Input('upload_detector', 'contents'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def load_detector(json_f):
    if json_f is None:
        raise dash.exceptions.PreventUpdate()
    try:
        det_dict = sf.process_dcc_upload_json(contents=json_f)
    except JSONDecodeError:
        return (
            True, read_json_error.header, read_json_error.body), None, no_upd, no_upd, no_upd, no_upd, no_upd, no_upd
    data = exp1.load_instrument_unit(det_dict, object_='detector')
    if data:
        if data[1] == 'Rectangle':
            return no_upd, None, no_upd, data[0], 'Rectangle', no_upd, data[0], 'Rectangle'
        else:
            return no_upd, None, data[0], no_upd, 'Circle', data[0], no_upd, 'Circle'
    else:
        return (True, load_det_error.header, load_det_error.body), None, no_upd, no_upd, no_upd, no_upd, no_upd, no_upd


@app.callback(
    Output('obstacle_div', 'children', allow_duplicate=True),
    Output('stored_obstacle_num', 'data', allow_duplicate=True),
    Input('add_obstacle_button', 'n_clicks'),
    State('obstacle_div', 'children'),
    State('stored_obstacle_num', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def generate_obstacle(n_cl, children, table_num):
    if n_cl == 0:
        raise dash.exceptions.PreventUpdate()
    new_div_table = apg.generate_obst_table(table_num)
    new_children = children.copy() + [new_div_table]
    table_num += 1
    return new_children, table_num


@app.callback(
    Output('obstacle_div', 'children'),
    Input({'type': 'obstacle_delete_div_button', 'index': ALL}, 'n_clicks'),
    State('obstacle_div', 'children'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def delete_obstacle(no, children):
    try:
        index_to_delete = no.index(1)
    except ValueError:
        raise dash.exceptions.PreventUpdate()
    children.pop(index_to_delete)
    return children


@app.callback(
    Output('set_wavelength_button', 'style'),
    Output('stored_wavelength_val', 'data', allow_duplicate=True),
    Input('set_wavelength_button', 'n_clicks'),
    State('wavelength_input', 'value'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def set_wavelength(n_clicks, wavelength):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    exp1.set_wavelength(wavelength=wavelength)
    return {'background-color': 'green'}, wavelength


@app.callback(
    Output('wavelength_input', 'value', allow_duplicate=True),
    Output('stored_wavelength_val', 'data', allow_duplicate=True),
    Output('upload_wavelength', 'contents'),
    Output('hidden_div_2', 'children', allow_duplicate=True),
    Input('upload_wavelength', 'contents'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def load_wavelength(contents):
    if contents is None:
        raise dash.exceptions.PreventUpdate()
    try:
        wavelength_dict = sf.process_dcc_upload_json((contents))
        result = exp1.load_instrument_unit(data_=wavelength_dict, object_='wavelength')
        if result:
            return result, result, None, no_upd
        else:
            return no_upd, no_upd, None, (True, load_wavelength_error.header, load_wavelength_error.body)
    except JSONDecodeError:
        return no_upd, no_upd, None, (True, read_json_error.header, read_json_error.body)


@app.callback(
    Output('download_wavelength', 'data'),
    Input('download_wavelength_btn', 'n_clicks')
)
@mylogger(level='DEBUG')
def download_wavelength(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()

    data_json = exp1.json_export('wavelength')
    return dict(content=data_json, filename='wavelength.json')


@app.callback(
    Output('download_goniometer', 'data'),
    Input('download_goniometer_btn', 'n_clicks'),
    prevent_initial_call=True

)
@mylogger(level='DEBUG')
def download_goniometer(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    if not hasattr(exp1, 'axes_rotations'):
        raise dash.exceptions.PreventUpdate()
    data_json = exp1.json_export('goniometer')
    return dict(content=data_json, filename='goniometer.json')


@app.callback(
    Output('stored_runs_div', 'data', allow_duplicate=True),
    Output('hidden_div_2', 'children', allow_duplicate=True),
    Output('Goniometer_table', 'data', allow_duplicate=True),
    Output('stored_goniometer_table', 'data', allow_duplicate=True),
    Output('upload_goniometer', 'contents'),
    Input('upload_goniometer', 'contents'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def load_goniometer(contents):
    if contents is None:
        raise dash.exceptions.PreventUpdate()
    try:
        data_list = sf.process_dcc_upload_json(contents)
    except JSONDecodeError:
        return no_upd, (True, read_json_error.header, read_json_error.body), no_upd, no_upd, None
    data = exp1.load_instrument_unit(data_list, 'goniometer')
    if data:
        content_vars.real_axes = data[1]
        content_vars.axes_angles = data[2]
        return None, no_upd, data[0], data[0], None
    else:
        return no_upd, (True, load_goniometer_error.header, load_goniometer_error.body), no_upd, no_upd, None


@app.callback(
    Output('upload_collision_log', 'contents'),
    Output('hidden_div_2', 'children', allow_duplicate=True),
    Input('upload_collision_log', 'contents'),

    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def load_log_collision(json_f):
    if json_f is None:
        raise dash.exceptions.PreventUpdate()
    try:
        json_data = sf.process_dcc_upload_json(json_f)
    except JSONDecodeError:
        print('error collision')
        return None, (True, read_json_error.header, read_json_error.body)
    # TODO add error handling
    exp1.set_logic_collision(json_data)
    return None, no_upd


@app.callback(
    Output('stored_log_collision_check', 'data'),
    Input('collision_check', 'value'),

    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def collision_check(check):
    if check is None or check == list():
        exp1.check_logic_collision = False
    else:
        exp1.check_logic_collision = True
    return check


@mylogger(level='DEBUG')
def apply_goniometer(dicts_list, return_dict=True):
    goniometer_dict = sf.list_of_dicts_to_dict(dicts_list, {
        'axes_rotations': str,
        'axes_directions': list,
        'axes_real': list,
        'axes_angles': list,
        'axes_names': list,
    }, )

    exp1.set_goniometer(**goniometer_dict)
    content_vars.real_axes = goniometer_dict['axes_real']
    content_vars.axes_angles = goniometer_dict['axes_angles']
    exp1.delete_scan(all_=True)
    if return_dict:
        return goniometer_dict


@app.callback(
    Output('stored_anvil_normal_data', 'data'),
    Output('stored_anvil_aperture_data', 'data'),
    Output('hidden_div_2', 'children', allow_duplicate=True),
    Input('set_anvil_btn', 'n_clicks'),
    State('anvil_normal_vector_table', 'data'),
    State('anvil_aperture_input', 'value'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def set_anvil(n_clicks, normal_data, aperture_data):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    normal = sf.procces_anvil_normal_data(normal_data)
    try:
        exp1.set_diamond_anvil(aperture=aperture_data, anvil_normal=normal)
    except DiamondAnvilError as e:
        return no_upd, no_upd, (True, e.error_modal_content.header, e.error_modal_content.body)
    return normal_data, aperture_data, no_upd


@app.callback(
    Output('stored_anvil_calc_check', 'data'),
    Input('calc_anvil_check', 'value'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def check_anvil(check):
    if check is None or check == list():
        exp1.calc_anvil_flag = False
    else:
        exp1.calc_anvil_flag = True
    return check


@app.callback(
    Output('set_obstacles_button', 'style'),
    Output('stored_obstacles_div', 'data', allow_duplicate=True),
    Input('set_obstacles_button', 'n_clicks'),
    State({'type': 'obstacle_table', 'index': ALL}, 'data'),
    State('obstacle_div', 'children'),
    prevent_initial_call=True

)
@mylogger(level='DEBUG')
def set_obstacles(n_clicks, children, obst_div):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    extracted_obstacles = []
    for obst in children:
        extracted_obstacle = sf.remake_obstacle_dict(obst[0])
        extracted_obstacles.append(extracted_obstacle)
        obst_is_corr = sf.check_obstacle_dict(extracted_obstacle, False)
        if not obst_is_corr: return {'background-color': 'red'}, no_upd
    exp1.clear_obstacles()
    for obst in extracted_obstacles:
        rot = (obst['rotation_x'], obst['rotation_y'], obst['rotation_z'])
        obst.pop('rotation_x')
        obst.pop('rotation_y')
        obst.pop('rotation_z')
        exp1.add_obstacles(**obst, rot=rot)
    return {'background-color': 'green'}, obst_div


@app.callback(Output('stored_obstacles_div', 'data', allow_duplicate=True),
          Input('clear_obstacles_button', 'n_clicks'),
          prevent_initial_call=True
          )
@mylogger(level='DEBUG')
def delete_obstacles(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    exp1.clear_obstacles()
    return None


@app.callback(Output('download_obstacles', 'data'),
          Input('download_obstacles_btn', 'n_clicks'))
@mylogger(level='DEBUG')
def download_obstacles(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    data_json = exp1.json_export('obstacles')
    return dict(content=data_json, filename='obstacles_input.json')


@app.callback(
    Output('upload_obstacles', 'contents'),
    Output('obstacle_div', 'children', allow_duplicate=True),
    Output('stored_obstacle_num', 'data', allow_duplicate=True),
    Output('stored_obstacles_div', 'data', allow_duplicate=True),
    Output('hidden_div_2', 'children', allow_duplicate=True),
    Input('upload_obstacles', 'contents'),
    State('stored_obstacle_num', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def load_obstacles(json_f, table_num):
    if not json_f:
        raise dash.exceptions.PreventUpdate()
    try:
        json_data = sf.process_dcc_upload_json(json_f)
    except JSONDecodeError:
        return None, no_upd, no_upd, no_upd, (True, read_json_error.header, read_json_error.body)
    data = exp1.load_instrument_unit(json_data, 'obstacles', table_num)
    if not data:
        return None, no_upd, no_upd, no_upd, (True, load_obstacles_error.header, load_obstacles_error.body)
    else:
        return None, data[0], data[1], data[0], no_upd


@app.callback(
    Output('modal2', 'is_open'),
    Output('modal_header2', 'children'),
    Output('modal_body2', 'children'),
    Input('hidden_div_2', 'children')
)
@mylogger(level='DEBUG')
def raise_modal2(children):
    if children == list():
        raise dash.exceptions.PreventUpdate()
    return children[0], children[1], children[2]


@app.callback(
    Output('download_instrument', 'data'),
    Input('download_instrument_btn', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def download_instrument(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    data_json = exp1.json_export('instrument')
    return dict(content=data_json, filename='instrument.json')


@app.callback(
    Output('hidden_div_2', 'children', allow_duplicate=True),

    Output('circle_parameter_table', 'data', allow_duplicate=True),
    Output('stored_circle_parameters', 'data', allow_duplicate=True),
    Output('rectangle_parameter_table', 'data', allow_duplicate=True),
    Output('stored_rectangle_parameters', 'data', allow_duplicate=True),
    Output('Det_geom_dropdown', 'value', allow_duplicate=True),
    Output('stored_detector_dropdown', 'data', allow_duplicate=True),

    Output('wavelength_input', 'value', allow_duplicate=True),
    Output('stored_wavelength_val', 'data', allow_duplicate=True),

    Output('stored_runs_div', 'data', allow_duplicate=True),
    Output('Goniometer_table', 'data', allow_duplicate=True),
    Output('stored_goniometer_table', 'data', allow_duplicate=True),

    Output('obstacle_div', 'children', allow_duplicate=True),
    Output('stored_obstacles_div', 'data', allow_duplicate=True),


    Output('linked_obstacle_div', 'children', allow_duplicate=True),
    Output('stored_linked_obstacles_div', 'data', allow_duplicate=True),

    Output('stored_obstacle_num', 'data', allow_duplicate=True),

    Output('upload_instrument', 'contents', allow_duplicate=True),
    Input('upload_instrument', 'contents'),
    State('stored_obstacle_num', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def load_instrument(json_f, table_num):
    if not json_f:
        raise dash.exceptions.PreventUpdate()
    try:
        json_data = sf.process_dcc_upload_json(json_f)
    except JSONDecodeError:
        return (True, read_json_error.header, read_json_error.body), *([no_upd] * 16), None
    instrument_data = exp1.load_instrument(data_=json_data, table_num=table_num)
    results = []

    flags_loaded = sf.get_loaded_flags(instrument_data)
    if not set(flags_loaded.keys()) != {}:
        error_message = load_instrument_error.body
        error_mapping = {
            'goniometer': load_det_diameter_error.body,
            'obstacles': load_obstacles_error.body,
            'detector': load_det_error.body,
            'wavelength': load_wavelength_error.body
        }
        for flag, message in error_mapping.items():
            if flags_loaded.get(flag, None) is False:
                error_message += message + '\n'
        results.append((True, load_instrument_error.header, error_message))
    else:
        results.append(no_upd)

    if instrument_data.get('detector',None):
        if instrument_data['detector'][1] == 'Circle':
            results.extend([*[instrument_data['detector'][0]] * 2, *[no_upd] * 2, *['Circle'] * 2])
        else:
            results.extend([*[no_upd] * 2, *[instrument_data['detector'][0]] * 2, *['Rectangle'] * 2])
    else:
        results.extend([no_upd] * 6)

    if instrument_data.get('wavelength',None):
        results.extend([*[instrument_data['wavelength']] * 2])
    else:
        results.extend([no_upd] * 2)

    if instrument_data.get('goniometer',None):
        content_vars.real_axes = instrument_data['goniometer'][1]
        content_vars.axes_angles = instrument_data['goniometer'][2]
        results.extend([None, *[instrument_data['goniometer'][0]] * 2])
    else:
        results.extend([no_upd] * 3)

    if instrument_data.get('obstacles',None):
        results.extend([*[instrument_data['obstacles'][0]] * 2])
    else:
        results.extend([no_upd] * 2)

    if instrument_data.get('linked_obstacles',None):
        results.extend([*[instrument_data['linked_obstacles'][0]] * 2])
    else:
        results.extend([no_upd] * 2)

    if instrument_data.get('table_num', None):
        results.extend([instrument_data['table_num']])
    else:
        results.extend([no_upd])

    return *results, None


@app.callback(
    Output({'type': 'obstacle_table', 'index': MATCH}, 'hidden_columns'),
    Input({'type': 'obstacle_table', 'index': ALL}, 'data'),
    prevent_initial_call=True
)
def obst_table_hidden_control(data_):
    data_changed = ctx.triggered[0]['value'][0]
    table_index = sf.find_index_digit(tuple(data_changed.keys())[0])
    hidden_columns = sf.prepare_hidden_columns(table_index, data_changed)
    return hidden_columns


@app.callback(
    Output({'type': 'linked_obstacle_table', 'index': MATCH}, 'hidden_columns'),
    Input({'type': 'linked_obstacle_table', 'index': ALL}, 'data'),
    prevent_initial_call=True
)
def obst_table_hidden_control_linked(data_):
    data_changed = ctx.triggered[0]['value'][0]
    table_index = sf.find_index_digit(tuple(data_changed.keys())[0])
    hidden_columns = sf.prepare_hidden_columns(table_index, data_changed)
    return hidden_columns


@app.callback(
    Output('linked_obstacle_div', 'children', allow_duplicate=True),
    Output('stored_obstacle_num', 'data', allow_duplicate=True),
    Input('add_linked_obstacle_button', 'n_clicks'),
    State('linked_obstacle_div', 'children'),
    State('stored_obstacle_num', 'data'),
    prevent_initial_call=True

)
def generate_linked_obstacle(n_cl, children, table_num):
    if n_cl == 0:
        raise dash.exceptions.PreventUpdate()
    if not exp1.axes_rotations:
        raise dash.exceptions.PreventUpdate()
    axes_dict = {key: val for key, val in enumerate(exp1.axes_names)}
    new_div_table = apg.generate_obst_table(table_num, linked=True, axes_dict=axes_dict)
    new_children = children.copy() + [new_div_table]
    table_num += 1
    return new_children, table_num


@app.callback(Output('stored_linked_obstacles_div', 'data', allow_duplicate=True),
          Input('clear_linked_obstacles_button', 'n_clicks'),
          prevent_initial_call=True
          )
@mylogger(level='DEBUG')
def delete_linked_obstacles(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    exp1.clear_linked_obstacles()
    return None


@app.callback(
    Output('set_linked_obstacles_button', 'style'),
    Output('stored_linked_obstacles_div', 'data', allow_duplicate=True),
    Input('set_linked_obstacles_button', 'n_clicks'),
    State({'type': 'linked_obstacle_table', 'index': ALL}, 'data'),
    State('linked_obstacle_div', 'children'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def set_linked_obstacles(n_clicks, children, obst_div):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    extracted_obstacles = []
    for obst in children:
        extracted_obstacle = sf.remake_obstacle_dict(obst[0])
        extracted_obstacles.append(extracted_obstacle)
        obst_is_corr = sf.check_obstacle_dict(extracted_obstacle, True)
        if not obst_is_corr: return {'background-color': 'red'}, no_upd
    exp1.clear_linked_obstacles()
    for obst in extracted_obstacles:
        rot = (obst['rotation_x'], obst['rotation_y'], obst['rotation_z'])
        obst.pop('rotation_x')
        obst.pop('rotation_y')
        obst.pop('rotation_z')
        exp1.add_linked_obstacle(**obst, rot=rot)
    return {'background-color': 'green'}, obst_div

@app.callback(Output('download_linked_obstacles', 'data'),
          Input('download_linked_obstacles_btn', 'n_clicks'))
@mylogger(level='DEBUG')
def download_linked_obstacles(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    data_json = exp1.json_export('linked_obstacles')
    return dict(content=data_json, filename='linked_obstacles_input.json')

@app.callback(
    Output('upload_linked_obstacles', 'contents'),
    Output('linked_obstacle_div', 'children', allow_duplicate=True),
    Output('stored_obstacle_num', 'data', allow_duplicate=True),
    Output('stored_linked_obstacles_div', 'data', allow_duplicate=True),
    Output('hidden_div_2', 'children', allow_duplicate=True),
    Input('upload_linked_obstacles', 'contents'),
    State('stored_obstacle_num', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def load_linked_obstacles(json_f, table_num):
    if not json_f:
        raise dash.exceptions.PreventUpdate()
    try:
        json_data = sf.process_dcc_upload_json(json_f)
    except JSONDecodeError:
        return None, no_upd, no_upd, no_upd, (True, read_json_error.header, read_json_error.body)
    data = exp1.load_instrument_unit(json_data, 'linked_obstacles', table_num)
    if not data:
        return None, no_upd, no_upd, no_upd, (True, load_obstacles_error.header, load_obstacles_error.body)
    else:
        return None, data[0], data[1], data[0], no_upd