from dash import html, dcc, dash_table, Input, Output, State, no_update as no_upd
from dash.dash_table.Format import Format, Scheme
import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc


from app import app
from global_state import content_vars, exp1
import services.service_functions as sf
from logger.my_logger import mylogger


layout = html.Div([
    html.Div([
        html.H2('UB_matrix Å−1'),
        html.Button(
            '?',
            id='test_button',
            n_clicks=0,
        ),
        dash_table.DataTable(
            fill_width=False,
            id='UB_table',
            data=content_vars.df_matr,
            columns=[{
                'id': 'a*',
                'name': 'a*',
                'type': 'numeric',
                'format': Format(
                    precision=6,
                    scheme=Scheme.fixed,
                ),
            }, {
                'id': 'b*',
                'name': 'b*',
                'type': 'numeric',
                'format': Format(
                    precision=6,
                    scheme=Scheme.fixed,
                ),
            }, {
                'id': 'c*',
                'name': 'c*',
                'type': 'numeric',
                'format': Format(
                    precision=6,
                    scheme=Scheme.fixed,
                ),
            }, ],
            editable=True,
            style_table={
                'overflowY': 'auto'
            },
            style_cell={
                'height': '0px',
                'maxHeight': '1px',
                'maxWidth': 60,
                'width': 60,
                'minWidth': 60
            },

        ),
        html.Button(
            children='Set cell by UB matrix',
            style={
                'verticalAlign': 'middle'
            },
            id='Set_UB_button',
            n_clicks=0,
            accessKey='U'

        ),
        dcc.Upload(html.Button('Import UB', style={'display': 'inline-block'}), id='upload_ub',
                   accept='.p4p, .par, .txt',
                   multiple=False, max_size=5000000),

    ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '0vw',
            'margin-top': '0vw'},
        id='UB_set_matrix_div'
    ),

    html.Div([
        html.H2('Transform matrix'),
        dash_table.DataTable(
            fill_width=False,
            id='Transform_matrix',
            data=content_vars.df_matr_transform,
            columns=[{
                'id': 'TM1',
                'name': 'TM1',
                'type': 'numeric',
                'format': Format(
                    precision=6,
                    scheme=Scheme.fixed,
                ),
            }, {
                'id': 'TM2',
                'name': 'TM2',
                'type': 'numeric',
                'format': Format(
                    precision=6,
                    scheme=Scheme.fixed,
                ),
            }, {
                'id': 'TM3',
                'name': 'TM3',
                'type': 'numeric',
                'format': Format(
                    precision=6,
                    scheme=Scheme.fixed,
                ),
            }, ],
            editable=True,
            style_header={
                'display': 'none'
            },
            style_table={
            },
            style_cell={
                'height': '0px',
                'maxHeight': '1px',
                'maxWidth': 60,
                'width': 60,
                'minWidth': 60
            },

        ),
        html.Button(
            children='Apply trans. matrix',
            style={
                'verticalAlign': 'middle'
            },
            id='Apply_transformation',
            n_clicks=0,

        ),
    ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '0vw',
            'margin-top': '0vw'},
        id='Transform_matrix_div'
    ),

    html.Div(
        [
            html.H2('Bravis cell parameters and orientation'),
            dash_table.DataTable(
                fill_width=False,
                id='Cell_parameters',
                data=content_vars.df_parameters,
                columns=[{
                    'id': 'a',
                    'name': 'a',
                    'type': 'numeric',
                    'format': Format(
                        precision=4,
                        scheme=Scheme.fixed,
                    ),
                }, {
                    'id': 'b',
                    'name': 'b',
                    'type': 'numeric',
                    'format': Format(
                        precision=6,
                        scheme=Scheme.fixed,
                    ),
                }, {
                    'id': 'c',
                    'name': 'c',
                    'type': 'numeric',
                    'format': Format(
                        precision=6,
                        scheme=Scheme.fixed,
                    ),
                },
                    {
                        'id': 'alpha',
                        'name': 'α',
                        'type': 'numeric',
                        'format': Format(
                            precision=4,
                            scheme=Scheme.fixed,
                        ),
                    },
                    {
                        'id': 'beta',
                        'name': 'β',
                        'type': 'numeric',
                        'format': Format(
                            precision=4,
                            scheme=Scheme.fixed,
                        ),
                    },
                    {
                        'id': 'gamma',
                        'name': 'γ',
                        'type': 'numeric',
                        'format': Format(
                            precision=4,
                            scheme=Scheme.fixed,
                        ),
                    },
                    {
                        'id': 'omega',
                        'name': '(+)ω',
                        'type': 'numeric',
                        'format': Format(
                            precision=4,
                            scheme=Scheme.fixed,
                        ),
                    },
                    {
                        'id': 'chi',
                        'name': '(-)χ',
                        'type': 'numeric',
                        'format': Format(
                            precision=4,
                            scheme=Scheme.fixed,
                        ),
                    },
                    {
                        'id': 'phi',
                        'name': '(-)φ',
                        'type': 'numeric',
                        'format': Format(
                            precision=4,
                            scheme=Scheme.fixed,
                        ),
                    },
                ],
                editable=True,
                style_table={
                },
                style_cell={
                    'height': '0px',
                    'maxHeight': '1px',
                    'maxWidth': 55,
                    'width': 55,
                    'minWidth': 55
                }
            ),
            html.Button(
                children='Set cell by cell parameters and Euler angles',
                style={
                    'verticalAlign': 'middle'
                },
                id='set_cell_by_parameters'
            )
        ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '5vw',
            'margin-top': '0vw',
        }),

],
    style={
        'height': '700px'
    }
)



@app.callback(
    Output("UB_table", "data", allow_duplicate=True),
    Output("Cell_parameters", "data", allow_duplicate=True),
    Output("home_page_stored_flag", "data", allow_duplicate=True),
    Input("home_page_stored_flag", "data"),
    State('stored_UB_table', 'data'),
    State('stored_parameters_table', 'data'),

    prevent_initial_call=True)
@mylogger(level='DEBUG')
def get_stored_home_page_data(flag, UB_table, parameters_table):
    if not flag:
        raise dash.exceptions.PreventUpdate()
    output_list = list()
    if UB_table is not None:
        output_list += [UB_table, ]
    else:
        output_list += [no_upd, ]
    if parameters_table is not None:
        output_list += [parameters_table, ]
    else:
        output_list += [no_upd, ]
    output_list += [False, ]
    return output_list


@app.callback(
    Output('UB_table', 'data', allow_duplicate=True),
    Output('stored_UB_table', 'data', allow_duplicate=True),
    Output('stored_parameters_table', 'data', allow_duplicate=True),
    Input('set_cell_by_parameters', 'n_clicks'),
    State('Cell_parameters', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def set_cell_by_parameters_matr(n_clicks, data):
    if n_clicks == 0 or n_clicks is None:
        raise dash.exceptions.PreventUpdate()
    prm = data[0]
    exp1.set_cell(parameters=(prm['a'], prm['b'], prm['c'], prm['alpha'], prm['beta'], prm['gamma']),
                  om_chi_phi=(prm['omega'], prm['chi'], prm['phi']))
    ub_matr = exp1.cell.orient_matx.copy()
    matr = pd.DataFrame({
        'a*': [ub_matr[0, 0], ub_matr[1, 0], ub_matr[2, 0]],
        'b*': [ub_matr[0, 1], ub_matr[1, 1], ub_matr[2, 1]],
        'c*': [ub_matr[0, 2], ub_matr[1, 2], ub_matr[2, 2]]
    }
    ).to_dict('records')
    content_vars.df_parameters = data
    content_vars.df_matr = matr
    return matr, matr, data


@app.callback(
    Output('Cell_parameters', 'data', allow_duplicate=True),
    Output('stored_UB_table', 'data', allow_duplicate=True),
    Output('stored_parameters_table', 'data', allow_duplicate=True),
    Input('Set_UB_button', 'n_clicks'),
    State('UB_table', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def set_cell_by_UB_matr(n_clicks, data):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    a_ = np.array([data[0]['a*'], data[1]['a*'], data[2]['a*']]).reshape(-1, 1)
    b_ = np.array([data[0]['b*'], data[1]['b*'], data[2]['b*']]).reshape(-1, 1)
    c_ = np.array([data[0]['c*'], data[1]['c*'], data[2]['c*']]).reshape(-1, 1)
    ub_array = np.hstack((a_, b_, c_))
    exp1.set_cell(matr=ub_array)
    parameters_ = exp1.cell.parameters.copy()
    parameters = pd.DataFrame({'a': [parameters_[0], ], 'b': [parameters_[1], ], 'c': [parameters_[2], ],
                               'alpha': [parameters_[3], ], 'beta': [parameters_[4], ], 'gamma': [parameters_[5], ],
                               'omega': [0, ],
                               'chi': [0, ], 'phi': [0, ]}).to_dict('records')
    content_vars.df_parameters = parameters
    content_vars.df_matr = data
    return parameters, data, parameters


@app.callback(
    Output('UB_table', 'data', allow_duplicate=True),
    Output('Cell_parameters', 'data', allow_duplicate=True),
    Output('stored_UB_table', 'data', allow_duplicate=True),
    Output('stored_parameters_table', 'data', allow_duplicate=True),
    Input('upload_ub', 'contents'),
    State('upload_ub', 'filename'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def load_ub_matrix(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate()
    data_str = sf.process_dcc_upload_file_to_str(contents)
    ub_array = None
    if filename.endswith('.p4p'):
        ub_array = sf.parse_p4p_for_UB(data_str)
    elif filename.endswith('.par'):
        ub_array = sf.parse_par_for_UB(data_str)
    if ub_array is not None:
        exp1.set_cell(matr=ub_array)
        parameters_ = exp1.cell.parameters.copy()
        parameters = pd.DataFrame({'a': [parameters_[0], ], 'b': [parameters_[1], ], 'c': [parameters_[2], ],
                                   'alpha': [parameters_[3], ], 'beta': [parameters_[4], ], 'gamma': [parameters_[5], ],
                                   'omega': [0, ],
                                   'chi': [0, ], 'phi': [0, ]}).to_dict('records')
        content_vars.df_parameters = parameters
        matr = pd.DataFrame({
            'a*': [ub_array[0, 0], ub_array[1, 0], ub_array[2, 0]],
            'b*': [ub_array[0, 1], ub_array[1, 1], ub_array[2, 1]],
            'c*': [ub_array[0, 2], ub_array[1, 2], ub_array[2, 2]]
        }
        ).to_dict('records')
        content_vars.df_matr = matr
        return matr, parameters, matr, parameters
    return no_upd, no_upd, no_upd, no_upd


@app.callback(
    Output('UB_table', 'data', allow_duplicate=True),
    Input('Apply_transformation', 'n_clicks'),
    State('UB_table', 'data'),
    State('Transform_matrix', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def apply_transformation_matrix(n_clicks, ub_matrix, t_matrix):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()

    ub_matrix = np.array(pd.DataFrame.from_records(ub_matrix))
    t_matrix = np.array(pd.DataFrame.from_records(t_matrix))
    matr = np.matmul(t_matrix, ub_matrix)
    ub_matrix_out = pd.DataFrame({
        'a*': [matr[0, 0], matr[1, 0], matr[2, 0]],
        'b*': [matr[0, 1], matr[1, 1], matr[2, 1]],
        'c*': [matr[0, 2], matr[1, 2], matr[2, 2]]
    }
    ).to_dict('records')
    return ub_matrix_out