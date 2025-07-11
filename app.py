from json import JSONDecodeError
from services.exceptions.exceptions import *
from dash import Dash, dash_table, html, dcc, callback, Input, Output, State, ALL, Patch, MATCH, ctx
from dash import no_update as no_upd
from dash.dash_table.Format import Format, Scheme
import pandas as pd
import dash
from os import listdir
import services.service_functions as sf
from assets.modals_content import *
from experiment.experiment import Experiment
import numpy as np
import dash_bootstrap_components as dbc
from symmetry.pointsymmetry import PG_KEYS, CENTRINGS
from dash_extensions import Keyboard
import assets.app_gens as apg
import json
from logger.my_logger import mylogger
import os
import copy
from sample.sample import Sample

css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style_main.css')
goniometer_models_path = os.path.join(os.path.dirname(__file__), 'instruments', 'goniometers')

exp1 = Experiment()

app = Dash(external_stylesheets=[
    dbc.themes.SIMPLEX,
    css_path,
], suppress_callback_exceptions=True)

point_groups = [*PG_KEYS.keys()]
centrings = list(CENTRINGS)

goniometers = [i[:-5] for i in listdir(goniometer_models_path) if i.endswith('.json')]

LOG_FILE = "app.log"
ZIPPED_LOG_FILE = "app_log.zip"


class Content_variables():
    def __init__(self):
        self.df_matr = pd.DataFrame(
            {'a*': [1., 0., 0.],
             'b*': [0., 1., 0.],
             'c*': [0., 0., 1.]
             }).to_dict('records')
        self.df_parameters = pd.DataFrame({
            'a': [10., ],
            'b': [10., ],
            'c': [10., ],
            'alpha': [90., ],
            'beta': [90., ],
            'gamma': [90., ],
            'omega': [0., ],
            'chi': [0., ],
            'phi': [0., ]}).to_dict('records')
        self.df_matr_transform = pd.DataFrame(
            {'TM1': [1., 0., 0.],
             'TM2': [0., 1., 0.],
             'TM3': [0., 0., 1.]
             }).to_dict('records')
        self.goniometer_table = pd.DataFrame(
            columns=('rotation', 'direction', 'name', 'real', 'angle')).to_dict('records')
        self.detector_geometry = pd.DataFrame(np.array([['', '', '', '']]),
                                              columns=('d_geometry', 'height', 'width', 'diameter')).to_dict('records')
        self.detector_geometry = pd.DataFrame(np.array([['']]), columns=('geometry',)).to_dict('records')
        self.detector_geometry_rectangle_parameters = pd.DataFrame(np.array([['', '']]),
                                                                   columns=('height_prm', 'width_prm')).to_dict(
            'records')
        self.detector_geometry_circle_parameters = pd.DataFrame(np.array([['']]),
                                                                columns=('diameter_prm',)).to_dict('records')
        self.complex_detector_parameters = pd.DataFrame(np.array([['', '', '', '']]),
                                                        columns=('rows_prm', 'columns_prm', 'row_spacing_prm',
                                                                 'column_spacing_prm')).to_dict('records')
        self.obstacles_parameters = pd.DataFrame(
            {'ditsance': [10, ], 'geometry': ['circle', ], 'orientation': ['normal', ],
             'rotations': [(0, 0, 0), ], 'lin_prm': [(0,)], 'displ y,z': [(0, 0)]}).to_dict('records')
        self.active_space_fig = None
        self.real_axes = None
        self.axes_angles = None


content_vars = Content_variables()

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#5C5C5C",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

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

sidebar = html.Div(
    [
        html.H1("Difs", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("UB Matix", href="/", active="exact"),
                dbc.NavLink("Instrumental model", href="/page-1", active="exact"),
                dbc.NavLink("Strategy", href="/page-2", active="exact"),
                dbc.NavLink("Calculation", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

first_page = html.Div([
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

second_page = html.Div([
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

modal3 = html.Div([
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(children='Header', id='modal_header3')),
        dbc.ModalBody(children='Body', id='modal_body3',
                      style={
                          'whiteSpace': 'pre-wrap',
                          'tabSize': '4',
                          'fontFamily': 'monospace'
                      }
                      )
    ],
        id='modal3',
        size='lg',
        is_open=False
    )
])

third_page = html.Div([

    html.Div(list(), id='hidden_div_3', style={'display': 'none'}),
    modal3,
    html.H4('Runs'),
    html.Div(
        [html.Button(
            'Add run',
            id='add_run_btn',
            n_clicks=0),
            html.Button(
                'Delete runs',
                id='clear_runs_btn',
                n_clicks=0),
            html.Button(
                'Set runs',
                id='set_runs_btn',
                n_clicks=0),
            html.Button(
                'Export runs',
                id='download_runs_btn',
                n_clicks=0),
            dcc.Download(id='download_runs'),
            dcc.Upload(html.Button(
                'Import runs', ), id='upload_runs', accept='.json', multiple=False, max_size=1000000)

        ],

    ),

    html.Div(
        list(),
        id='runs_div'
    )

])

modal4 = html.Div([
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(children='Header', id='modal_header4')),
        dbc.ModalBody(children='Body', id='modal_body4',
                      style={
                          'whiteSpace': 'pre-wrap',
                          'tabSize': '4',
                          'fontFamily': 'monospace'
                      }
                      )
    ],
        id='modal4',
        size='lg',
        is_open=False
    )
]
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
fourth_page = html.Div([

    html.Div(list(), id='hidden_div_4', style={'display': 'none'}),
    modal4,
    html.Div([
        html.P('Point group'),
        dcc.Dropdown(
            options=point_groups,
            value='',
            id='point_group_selector',
            clearable=False)
    ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '0vw',
            'margin-top': '1vw',
            'width': '90px'}
    ),
    html.Div([
        html.P('Centring'),
        dcc.Dropdown(
            options=centrings,
            value='',
            id='centring_selector',
            clearable=False)
    ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '1vw',
            'margin-top': '1vw',
            'width': '70px'}
    ),
    html.Div([
        html.P('d range, Å'),
        dash_table.DataTable(
            data=pd.DataFrame({'min': 0.68, 'max': 10}, index=[0]).to_dict('records'),
            columns=[
                {'name': 'min', 'id': 'min', 'type': 'numeric', 'format': Format(
                    precision=2,
                    scheme=Scheme.fixed,
                )},
                {'name': 'max', 'id': 'max', 'type': 'numeric', 'format': Format(
                    precision=2,
                    scheme=Scheme.fixed,
                )},

            ],
            id='d_range_table',
            editable=True)
    ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '1vw',
            'margin-top': '0vw',
            'width': '100px'}
    ),
    html.Div([
        html.Button('Calculate',
                    id='calc_experiment_btn',
                    n_clicks=0)
    ],

        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '1vw',
            'margin-top': '3vw',
            'width': '100px'}
    ),
    html.Div([
        html.Button('Save as hkl',
                    id='download_data_hkl_btn',
                    n_clicks=0)
    ],

        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '1vw',
            'margin-top': '3vw',
            'width': '100px'}
    ),
    html.Div([
        html.Button('Sep unique and common',
                    id='separate_unique_common_btn',
                    n_clicks=0)
    ],

        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '1vw',
            'margin-top': '3vw',
            'width': '100px'}
    ),
    dcc.Download(id='download_data_hkl'),
    dcc.Upload([
        html.Button('Import hkl',
                    )
    ],
        style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '1vw',
            'margin-top': '3vw',
            'width': '100px'},
        id='upload_hkl', accept='.hkl', multiple=True, max_size=1_000_000_00
    ),
    html.Div([
        html.H5('Completeness %'),
        html.H5('', id='completeness_header')
    ],
    ),
    html.Div([
        html.Div([
            html.H6('Completeness %'),
            html.Div([
                dcc.Graph(
                    id='completeness_graph'
                )
            ],
                id='completeness_plot_div',
                className='plotlygraph'
            )
        ],
            style={'display': 'block',
                   'height': '500px',
                   'width': '1000px'}),
        html.Div([
            html.H6('Multiplicity'),
            html.Div([dcc.Graph(
                id='multiplicity_graph',
                className='plotlygraph'

            )],
                id='multiplicity_plot_div')
        ],
            style={'display': 'block',
                   'height': '500px',
                   'width': '1000px'}
        ),
        html.Div([
            html.H6('Redundancy'),
            html.Div([dcc.Graph(
                id='redundancy_graph',
                className='plotlygraph'

            )],
                id='redundancy_plot_div')
        ],
            style={'display': 'block',
                   'height': '500px',
                   'width': '1000px'}
        ),
        html.Div([
            html.H6('Cumulative completeness'),
            html.Div(
                children=dcc.Graph(id='cumulative_plot_graph')
                ,
                id='cumulative_plot_container'
            ),
            html.Button('Update', id='update_cumulative_data_btn', n_clicks=0),
            html.Div([
                html.Div(apg.generate_empty_dag_for_cumulative_completeness(id_='completeness_dag'),
                         ),

            ],
                id='div_ag_container'
            ),

            html.Div(

                [html.Button('Sort and calc', id='calc_cumulative_reorder_comp_plot_btn', n_clicks=0),
                 html.Button('Calc', id='calc_cumulative_ordered_comp_plot_btn', n_clicks=0),
                 html.Div([
                     html.H4('Calc d range, Å, '),
                     apg.get_range_dag(id_='cumulative_range')
                 ]
                     , id='div_ag_parameters_container'),
                 ]
            ),

        ])
    ],
        id='div_plots_container',
        style={'display': 'block'}
    ),
    html.Div([
        Keyboard(captureKeys=['1', '2', '3'], id='move_keyboard'),
        Keyboard(captureKeys=['4', '5', '6'], id='resize_keyboard'),
        Keyboard(captureKeys=['7', '8', '9'], id='rotate_keyboard'),
        Keyboard(captureKeys=['Enter'], id='to_active_keyboard'),
        html.H6('Reciprocal space viewer'),
        html.Button('known space', id='known_space_button', n_clicks=0),
        html.Button('known hkl', id='known_hkl_button', n_clicks=0),
        html.Button('known independent_hkl', 'known_hkl_orig_button', n_clicks=0),
        html.Div(dcc.Graph(id='rec_space_viewer_graph', style={
            'width': '1000px',
            'height': '1000px'},
                           className='plotlygraph'

                           ),
                 id='rec_space_viewer_div',
                 style={'height': '1000px', 'width': '1000px'}
                 ),
        html.Div([
            html.Button('Delete active', id='delete_active_button', n_clicks=0),
            html.Button('Reset active', id='reset_active_button', n_clicks=0),
            html.Button('Inverse active', id='inverse_active_button', n_clicks=0),
            html.Button('Unknown to active', id='unknwn_to_active_button', n_clicks=0),
            html.Button('Select active', id='sel_points_button', n_clicks=0),
            html.Button('Sphere selection', id='sphere_selection_button', n_clicks=0),
            html.Button('Box selection', id='cuboid_selection_button', n_clicks=0),
            html.Button('Cone selection', id='cone_selection_button', n_clicks=0),

        ])
    ],
        style={'height': '1500px', 'width': '1500px'}
    ),
    html.Div([
        html.Div([html.H6('Diffraction map'),
                  dcc.Dropdown(options=[
                      '3d map', '2d map', '1d map'
                  ],
                      id='dropdown_map_switcher',
                      clearable=False,
                      value='',
                      style={'width': '200px'},

                  ),
                  apg.get_diff_map_detector(id_='diff_map_detector'),
                  html.Div(id='map_input_container'),
                  html.Button('map for selected', id='map_selected_button', n_clicks=0), ]
                 , style={'display': 'inline-block', }),

        html.Div([dcc.Graph(id='diffraction_map_graph', style={
            'width': '1000px',
            'height': '1000px'},
                            className='plotlygraph'
                            )
                  ],
                 id='diffraction_map_graph_div'
                 )
    ],
        id='diffraction_map_div',
        style={'display': 'block'},
    ),
    html.Div(
        [
            html.H6('Calc rotation'),

            html.P('none', id='diff_map_workaround_P', style={'display': 'none'}),
            html.P('... reflections at selected range', id='ref_at_selected_range'),
            html.P('... unique reflections at selected range', id='u_ref_at_selected_range'),
            html.P('... independent reflections at selected range', id='o_ref_at_selected_range')
        ]
    )

]
)
STORE_TYPE = 'memory'
app.layout = html.Div([dcc.Location(id="url"),
                       sidebar,
                       content,
                       dcc.Store(data=None, id='old_url', storage_type=STORE_TYPE),
                       dcc.Store(data=False, id='rec_space_viewer_graph_has_data', storage_type=STORE_TYPE),
                       # home page,
                       dcc.Store(data=False, id='home_page_stored_flag', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_UB_table', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_parameters_table', storage_type=STORE_TYPE),
                       # page-1 parameters page
                       dcc.Store(data=list(), id='stored_log_collision_check', storage_type=STORE_TYPE),
                       dcc.Store(data=False, id='page-1_stored_flag', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_wavelength_val', storage_type=STORE_TYPE, ),
                       dcc.Store(data=None, id='stored_goniometer_table', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_goniometer_dropdown', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_detector_dropdown', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_detector_check_complex', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_circle_parameters', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_rectangle_parameters', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_obstacles_div', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_linked_obstacles_div', storage_type=STORE_TYPE),
                       dcc.Store(data=0, id='stored_obstacle_num', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_anvil_normal_data', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_anvil_aperture_data', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_anvil_calc_check', storage_type=STORE_TYPE),
                       # page-2 runs page
                       dcc.Store(data=False, id='page-2_stored_flag', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_runs_div', storage_type=STORE_TYPE),
                       dcc.Store(data=0, id='stored_runs_num', storage_type=STORE_TYPE),
                       # page-3 calc page
                       dcc.Store(data=False, id='page-3_stored_flag', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_centring_dropdown', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_d_range_table', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_point_group_dropdown', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_completeness_val', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_completeness_plot', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_multiplicity_plot', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_redundancy_plot', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_rec_space', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_diffraction_map', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_section_vals', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_cumulative_dag_data', storage_type=STORE_TYPE),
                       dcc.Store(data=None, id='stored_cumulative_plot', storage_type=STORE_TYPE),
                       ])


@app.callback(
    Output("page-content", "children"),
    Output("home_page_stored_flag", "data"),
    Output("page-1_stored_flag", "data"),
    Output("page-2_stored_flag", "data"),
    Output("page-3_stored_flag", "data"),
    Output('old_url', 'data'),
    Output('stored_rec_space', 'data'),
    Input("url", "pathname"),
    State("old_url", "data"),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def render_page_content(pathname, old_url):
    rec_fig = get_rec_figure(old_url=old_url)
    if pathname == "/":
        return first_page, True, no_upd, no_upd, no_upd, pathname, rec_fig
    elif pathname == "/page-1":
        return second_page, no_upd, True, no_upd, no_upd, pathname, rec_fig
    elif pathname == "/page-2":
        return third_page, no_upd, no_upd, True, no_upd, pathname, rec_fig
    elif pathname == "/page-3":
        return fourth_page, no_upd, no_upd, no_upd, True, pathname, rec_fig
    return html.Div(
        [
            html.H2("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3", ), no_upd, no_upd, no_upd, no_upd, pathname, rec_fig


def get_rec_figure(old_url):
    if old_url != '/page-3' or old_url is None or content_vars.active_space_fig is None:
        return no_upd
    else:
        return content_vars.active_space_fig.fig


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


# stored_linked_obstacles_div
# linked_obstacle_div

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
    Output('runs_div', 'children', allow_duplicate=True),
    Output("page-2_stored_flag", "data", allow_duplicate=True),
    Input("page-2_stored_flag", "data"),
    State('stored_runs_div', 'data'),
    prevent_initial_call=True)
def get_stored_page_2_data(flag, runs):
    if not flag:
        raise dash.exceptions.PreventUpdate()
    output_list = [if_val_None_return_no_upd_else_return(runs), False]
    return output_list


@app.callback(
    Output('centring_selector', 'value'),
    Output('d_range_table', 'data'),
    Output('point_group_selector', 'value'),
    Output('completeness_header', 'children', allow_duplicate=True),
    Output('completeness_graph', 'figure', allow_duplicate=True),
    Output('multiplicity_graph', 'figure', allow_duplicate=True),
    Output('redundancy_graph', 'figure', allow_duplicate=True),
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Output('completeness_dag', 'rowData'),
    Output('cumulative_plot_graph', 'figure'),
    Output("page-3_stored_flag", "data", allow_duplicate=True),
    Input("page-3_stored_flag", "data"),
    State('stored_centring_dropdown', 'data'),
    State('stored_d_range_table', 'data'),
    State('stored_point_group_dropdown', 'data'),
    State('stored_completeness_val', 'data'),
    State('stored_completeness_plot', 'data'),
    State('stored_multiplicity_plot', 'data'),
    State('stored_redundancy_plot', 'data'),
    State('stored_rec_space', 'data'),
    State('stored_cumulative_dag_data', 'data'),
    State('stored_cumulative_plot', 'data'),
    prevent_initial_call=True)
def get_stored_page_3_data(flag, centring, d_range, pg, comp_val, comp_plot, mult_plot, red_plot, rec_space, dag_cum,
                           cum_plot):
    if not flag:
        raise dash.exceptions.PreventUpdate()
    storage_data_list = [centring, d_range, pg, comp_val, comp_plot, mult_plot, red_plot, rec_space, dag_cum, cum_plot]
    output_list = [if_val_None_return_no_upd_else_return(val) for val in storage_data_list]
    output_list += [False, ]
    return output_list


def if_val_None_return_no_upd_else_return(val):
    if val is None:
        return no_upd
    else:
        return val


@callback(
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
    prm_ornt = np.array(pd.DataFrame.from_records(data)).reshape(-1)
    exp1.set_cell(parameters=(prm_ornt[0], prm_ornt[1], prm_ornt[2], prm_ornt[3], prm_ornt[4], prm_ornt[5]),
                  om_chi_phi=(prm_ornt[6], prm_ornt[7], prm_ornt[8]))
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


@callback(
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


@callback(
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


@callback(
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


@callback(
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


@callback(
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


@callback(Output('stored_runs_div', 'data', allow_duplicate=True),
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


@callback(
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


@callback(
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


@callback(
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


@callback(
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


@callback(
    Output('download_detector', 'data'),
    Input('download_detector_btn', 'n_clicks'),
)
@mylogger(level='DEBUG')
def download_detector(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    json_data = exp1.json_export('detector')
    return dict(content=json_data, filename='detector_input.json')


@callback(
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


@callback(
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


@callback(
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


@callback(
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


@callback(
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


@callback(
    Output('download_wavelength', 'data'),
    Input('download_wavelength_btn', 'n_clicks')
)
@mylogger(level='DEBUG')
def download_wavelength(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()

    data_json = exp1.json_export('wavelength')
    return dict(content=data_json, filename='wavelength.json')


@callback(
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


@callback(
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


@callback(
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
        return None, (True, read_json_error.header, read_json_error.body)
    # TODO add error handling
    exp1.set_logic_collision(json_data)
    return None, no_upd


@callback(
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


@callback(
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


@callback(
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

    # extracted_obstacles = []
    # for obst in children:
    #     extracted_obstacle = sf.remake_obstacle_dict(obst[0])
    #     extracted_obstacles.append(extracted_obstacle)
    #     obst_is_corr = sf.check_obstacle_dict(extracted_obstacle, True)
    #     if not obst_is_corr: return {'background-color': 'red'}, no_upd
    # for obst in extracted_obstacles:
@callback(
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


@callback(Output('stored_obstacles_div', 'data', allow_duplicate=True),
          Input('clear_obstacles_button', 'n_clicks'),
          prevent_initial_call=True
          )
@mylogger(level='DEBUG')
def delete_obstacles(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    exp1.clear_obstacles()
    return None


@callback(Output('download_obstacles', 'data'),
          Input('download_obstacles_btn', 'n_clicks'))
@mylogger(level='DEBUG')
def download_obstacles(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    data_json = exp1.json_export('obstacles')
    return dict(content=data_json, filename='obstacles_input.json')


@callback(
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


@callback(
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


@callback(
    Output('modal3', 'is_open'),
    Output('modal_header3', 'children'),
    Output('modal_body3', 'children'),
    Input('hidden_div_3', 'children')
)
@mylogger(level='DEBUG')
def raise_modal3(children):
    if children == list():
        raise dash.exceptions.PreventUpdate()
    return children[0], children[1], children[2]


@callback(
    Output('modal4', 'is_open'),
    Output('modal_header4', 'children'),
    Output('modal_body4', 'children'),
    Input('hidden_div_4', 'children')
)
@mylogger(level='DEBUG')
def raise_modal4(children):
    if children == list():
        raise dash.exceptions.PreventUpdate()
    return children[0], children[1], children[2]


@callback(
    Output('hidden_div_3', 'children', allow_duplicate=True),
    Output('runs_div', 'children', allow_duplicate=True),
    Output('stored_runs_num', 'data', allow_duplicate=True),
    Input('add_run_btn', 'n_clicks'),
    State('runs_div', 'children'),
    State('stored_runs_num', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def add_runs(n_cl, children, table_num):
    if n_cl == 0:
        raise dash.exceptions.PreventUpdate()
    try:
        rotations = exp1.axes_rotations

    except AttributeError:
        return list(
            (True, 'Add run warning', 'Before adding scans, enter the instrument model of the goniometer')), children
    new_children = children.copy()
    real_axes = content_vars.real_axes
    axes_angles = content_vars.axes_angles
    names = exp1.axes_names
    new_div_table = apg.gen_run_table(real_axes, axes_angles, rotations, names, table_num)
    table_num += 1
    new_children += [new_div_table, ]
    return list((False, '', '')), new_children, table_num


@callback(
    Output('runs_div', 'children'),
    Input({'type': 'run_delete_div_button', 'index': ALL}, 'n_clicks'),
    State('runs_div', 'children'),
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


@callback(
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


@callback(
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


@callback(
    Output('hidden_div_3', 'children', allow_duplicate=True),
    Output('set_runs_btn', 'style'),
    Output('stored_runs_div', 'data', allow_duplicate=True),
    Input('set_runs_btn', 'n_clicks'),
    State({'type': 'run_table', 'index': ALL}, 'data'),
    State('runs_div', 'children'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def set_runs(n_clicks, data, children):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    data_ = []
    for run in data:
        run = list(run[0].values())
        data_.append(run)
    try:
        data_ = sf.process_runs_to_dicts_list(data_)
    except:
        return (list((True, add_runs_empty_error.header, add_runs_empty_error.body)), {'background-color': 'red'},
                no_upd)
    if exp1.logic_collision and exp1.check_logic_collision:
        try:
            sf.check_collision(exp_inst=exp1, runs=data_)
        except CollisionError as e:
            return (list((True, e.error_modal_content.header, e.error_modal_content.body)), {'background-color': 'red'},
                    no_upd)

    for run in data_:
        try:
            sf.check_run_dict_temp(run)
        except RunsDictError as e:
            return (list((True, e.error_modal_content.header, e.error_modal_content.body)), {'background-color': 'red'},
                    no_upd)
    exp1.scans = list()
    for run in data_:
        exp1.add_scan(**run)
    return list((False, '', '')), {'background-color': 'green'}, children


@callback(Output('hidden_div_3', 'children', allow_duplicate=True),
          Output('stored_runs_div', 'data', allow_duplicate=True),
          Input('clear_runs_btn', 'n_clicks'),
          prevent_initial_call=True
          )
@mylogger(level='DEBUG')
def delete_runs(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    if exp1.scans is None:
        return list((True, delete_runs_empty_error.header, delete_runs_empty_error.body)), no_upd
    exp1.delete_scan(all_=True)
    return list((False, '', '')), None


@callback(
    Output('download_runs', 'data'),
    Output('hidden_div_3', 'children', allow_duplicate=True),
    Input('download_runs_btn', 'n_clicks'),
    prevent_initial_call=True

)
@mylogger(level='DEBUG')
def download_runs(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    if not exp1.goniometer_is_set():
        return no_upd, (True, download_runs_error_gon.header, download_runs_error_gon.body)
    if not exp1.runs_are_set():
        return no_upd, (True, download_runs_error_runs.header, download_runs_error_runs.body)
    data_json = exp1.json_export('runs')
    return dict(content=data_json, filename='runs.json'), no_upd


@callback(
    Output('upload_runs', 'contents'),
    Output('hidden_div_3', 'children', allow_duplicate=True),
    Output('stored_runs_num', 'data', allow_duplicate=True),
    Output('runs_div', 'children', allow_duplicate=True),
    Output('stored_runs_div', 'data', allow_duplicate=True),
    Input('upload_runs', 'contents'),
    State('stored_runs_num', 'data'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def load_runs(contents, table_num):
    if contents is None:
        raise dash.exceptions.PreventUpdate()
    try:
        data_dict = sf.process_dcc_upload_json(contents)
    except JSONDecodeError:
        return None, (True, read_json_error.header, read_json_error.body), no_upd, no_upd, None
    try:
        runs_tables, table_num = exp1.load_scans(data_dict, table_num=table_num)
    except (RunsDictError, InstrumentError, CollisionError) as e:
        return None, (True, e.error_modal_content.header, e.error_modal_content.body), no_upd, no_upd, no_upd
    return None, no_upd, table_num, *[runs_tables] * 2


@callback(
    Input('centring_selector', 'value'),
)
@mylogger(level='DEBUG')
def set_centring(value):
    if value == '':
        raise dash.exceptions.PreventUpdate()
    exp1.set_centring(centring=value)


@callback(
    Input('point_group_selector', 'value'),
)
@mylogger(level='DEBUG')
def set_centring(value):
    if value == '':
        raise dash.exceptions.PreventUpdate()
    exp1.set_pg(pg=value)


@callback(
    Output('completeness_header', 'children', allow_duplicate=True),
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Output('calc_experiment_btn', 'style'),
    Output('stored_centring_dropdown', 'data'),
    Output('stored_d_range_table', 'data'),
    Output('stored_point_group_dropdown', 'data'),
    Output('stored_completeness_val', 'data', allow_duplicate=True),
    Input('calc_experiment_btn', 'n_clicks'),
    State('d_range_table', 'data'),
    State('point_group_selector', 'value'),
    State('centring_selector', 'value'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG', log_args=True)
def calc_experiment(n_clicks, children
                    , pg, centring
                    ):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    d_range = tuple(pd.DataFrame.from_records(children).iloc[0])

    if exp1.centring is None:
        return 'error', list((True, calc_exp_centring_error.header, calc_exp_centring_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd

    if exp1.pg is None:
        return 'error', list((True, calc_exp_pg_error.header, calc_exp_pg_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd

    if d_range[0] > d_range[1]:
        return 'error', list((True, calc_exp_min_max_error.header, calc_exp_min_max_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd
    if exp1.wavelength is None:
        return 'error', list((True, calc_exp_wavelength_error.header, calc_exp_wavelength_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd
    if exp1.scans is None:
        return 'error', list((True, calc_exp_no_scans_error.header, calc_exp_no_scans_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd

    try:
        exp1.calc_experiment(d_range=d_range)
    except CollisionError as e:
        return (
            'error', list((True, e.error_modal_content.header, e.error_modal_content.body)),
            {'background-color': 'red'},
            no_upd,
            no_upd, no_upd, no_upd)

    completeness = f'{exp1.show_completeness():.2f}'

    if exp1.det_geometry is None:
        return completeness, list((True, calc_exp_no_det_warn.header, calc_exp_no_det_warn.body)), {
            'background-color': 'green'}, centring, children, pg, completeness
    return completeness, list((False, '', '')), {'background-color': 'green'}, centring, children, pg, completeness


@callback(Output('hidden_div_4', 'children', allow_duplicate=True),
          Input('separate_unique_common_btn', 'n_clicks'),
          prevent_initial_call=True
          )
def separate_uiq_comm(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    if not exp1.strategy_data_container.hasdata() or len(exp1.strategy_data_container.scan_data_containers) < 2:
        return (True, separate_unique_common_error.header, separate_unique_common_error.body)
    exp1.separate_unique_common()
    return no_upd


@callback(
    Output('download_data_hkl', 'data'),
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Input('download_data_hkl_btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_data_hkl(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    try:
        hkl_data_str = exp1.form_scan_data_as_hkl()
    except NoScanDataError as e:
        return no_upd, (True, e.error_modal_content.header, e.error_modal_content.body)
    return dict(content=hkl_data_str, filename='output.hkl'), no_upd


@callback(
    Output('completeness_header', 'children', allow_duplicate=True),
    Output('stored_completeness_val', 'data', allow_duplicate=True),
    Output('upload_hkl', 'contents'),
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Input('upload_hkl', 'contents'),
    prevent_initial_call=True

)
def load_hkl(contents):
    if contents is None:
        raise dash.exceptions.PreventUpdate()
    hkl_list = []
    for hkl in contents:
        try:
            hkl_list.append(sf.process_dcc_upload_file_to_str(hkl))
        except HKLFormatError as e:
            return no_upd, no_upd, None, (True, e.error_modal_content.header, e.error_modal_content.body)
    exp1.load_hkls(hkl_list)
    completeness = f'{exp1.show_completeness():.2f}'
    return completeness, completeness, None, no_upd


@callback(
    Output('completeness_graph', 'figure'),
    Output('multiplicity_graph', 'figure'),
    Output('redundancy_graph', 'figure'),
    Output('stored_completeness_plot', 'data'),
    Output('stored_multiplicity_plot', 'data'),
    Output('stored_redundancy_plot', 'data'),
    Input('completeness_header', 'children'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def calc_1d_plots(children):
    if children == '':
        raise dash.exceptions.PreventUpdate()
    if children == 'error':
        return None, None, None, None, None, None
    figs = exp1.generate_1d_result_plot()
    return figs[0], figs[1], figs[2], figs[0], figs[1], figs[2]


@callback(
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('known_space_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def show_known_space(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    if hasattr(exp1, 'known_space'):
        fig = copy.copy(exp1.known_space)
        content_vars.active_space_fig = sf.plotly_fig(fig)
        return list((False, '', '')), fig
    else:
        if exp1.strategy_data_container.hasdata():
            if exp1.cell.cell_vol > 64000:
                return list((True, show_rec_cell_volume_error.header, show_rec_cell_volume_error.body)), no_upd
            fig = copy.copy(exp1.generate_known_space_3d())
            content_vars.active_space_fig = sf.plotly_fig(fig)
            return list((False, '', '')), fig
        else:
            return list((True, show_rec_no_data_error.header, show_rec_no_data_error.body)), no_upd


@callback(
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('known_hkl_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def show_known_hkl(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()

    if hasattr(exp1, 'known_hkl'):
        fig = copy.copy(exp1.known_hkl)
        content_vars.active_space_fig = sf.plotly_fig(fig)
        return list((False, '', '')), fig
    else:
        if exp1.strategy_data_container.hasdata():
            if exp1.cell.cell_vol > 64000:
                return list((True, show_rec_cell_volume_error.header, show_rec_cell_volume_error.body)), no_upd
            fig = copy.copy(exp1.generate_known_hkl_3d())
            content_vars.active_space_fig = sf.plotly_fig(fig)
            return list((False, '', '')), fig
        else:
            return list((True, show_rec_no_data_error.header, show_rec_no_data_error.body)), no_upd


@callback(
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('known_hkl_orig_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def show_known_hkl_orig(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    if hasattr(exp1, 'known_hkl_orig'):
        fig = copy.copy(exp1.known_hkl_orig)
        content_vars.active_space_fig = sf.plotly_fig(fig)
        return list((False, '', '')), fig
    else:
        if exp1.strategy_data_container.hasdata():
            if exp1.cell.cell_vol > 64000:
                return list((True, show_rec_cell_volume_error.header, show_rec_cell_volume_error.body)), no_upd
            fig = copy.copy(exp1.generate_known_hkl_orig_3d())
            content_vars.active_space_fig = sf.plotly_fig(fig)

            return list((False, '', '')), fig
        else:
            if exp1.cell.cell_vol > 64000:
                return list((True, show_rec_cell_volume_error.header, show_rec_cell_volume_error.body)), no_upd


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('rec_space_viewer_graph', 'clickData'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def active_point(clickdata):
    point_data = clickdata['points'][0]
    content_vars.active_space_fig.add_point_to_active(trace_n=point_data['curveNumber'],
                                                      index_n=point_data['pointNumber'])
    patched_fig = Patch()
    patched_fig['data'][point_data['curveNumber']]['marker']['color'][point_data['pointNumber']] = \
        content_vars.active_space_fig.fig['data'][point_data['curveNumber']]['marker']['color'][
            point_data['pointNumber']]
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('delete_active_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def delete_active(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.delete_active()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig['data']
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('sphere_selection_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def add_selection_sphere(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.create_selection_sphere()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig['data']
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('cone_selection_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def add_selection_cone(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.create_selection_cone()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig['data']
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('cuboid_selection_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def create_cuboid(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.create_selection_cuboid()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig['data']
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('move_keyboard', 'n_keydowns'),
    State('move_keyboard', 'keydown'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def move_selection_fig(n_keys, key):
    key_ = key['key']
    altkey = key['altKey']
    if n_keys == 0 or key_ not in '123':
        raise dash.exceptions.PreventUpdate()
    if content_vars.active_space_fig is None:
        raise dash.exceptions.PreventUpdate()
    if 'selection' not in content_vars.active_space_fig.fig.data[-1]['name']:
        raise dash.exceptions.PreventUpdate()

    distance = 0.05
    if altkey: distance *= -1

    if key_ == '1':
        content_vars.active_space_fig.move_selection_figure('x', distance)
    elif key_ == '2':
        content_vars.active_space_fig.move_selection_figure('y', distance)
    elif key_ == '3':
        content_vars.active_space_fig.move_selection_figure('z', distance)
    patched_fig = Patch()
    patched_fig['data'][-1] = content_vars.active_space_fig.fig.data[-1]
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('resize_keyboard', 'n_keydowns'),
    State('resize_keyboard', 'keydown'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def resize_selection_fig(n_keys, key):
    key_ = key['key']
    altkey = key['altKey']
    if n_keys == 0 or key_ not in '456':
        raise dash.exceptions.PreventUpdate()
    if content_vars.active_space_fig is None:
        raise dash.exceptions.PreventUpdate()
    if 'selection' not in content_vars.active_space_fig.fig.data[-1]['name']:
        raise dash.exceptions.PreventUpdate()
    growth = 0.05
    if altkey: growth *= -1

    if key_ == '4':
        content_vars.active_space_fig.resize_selection_figure(axis='x', growth=growth)
    elif key_ == '5':
        content_vars.active_space_fig.resize_selection_figure(axis='y', growth=growth)
    elif key_ == '6':
        content_vars.active_space_fig.resize_selection_figure(axis='z', growth=growth)
    patched_fig = Patch()
    patched_fig['data'][-1] = content_vars.active_space_fig.fig.data[-1]
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('to_active_keyboard', 'n_keydowns'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def figure_to_active(n_keys):
    if n_keys == 0:
        raise dash.exceptions.PreventUpdate()
    if content_vars.active_space_fig is None:
        raise dash.exceptions.PreventUpdate()
    if 'selection' not in content_vars.active_space_fig.fig.data[-1]['name']:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.active_by_figure()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig.data
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('rotate_keyboard', 'n_keydowns'),
    State('rotate_keyboard', 'keydown'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def rotate_selection_fig(n_keys, key):
    key_ = key['key']
    altkey = key['altKey']
    if n_keys == 0:
        raise dash.exceptions.PreventUpdate()
    if content_vars.active_space_fig is None:
        raise dash.exceptions.PreventUpdate()
    if 'selection' not in content_vars.active_space_fig.fig.data[-1]['name']:
        raise dash.exceptions.PreventUpdate()

    angle = 5
    if altkey: angle *= -1
    if key_ == '7':
        content_vars.active_space_fig.rotate_sel_figure(axis='x', angle=angle)
    elif key_ == '8':
        content_vars.active_space_fig.rotate_sel_figure(axis='y', angle=angle)
    elif key_ == '9':
        content_vars.active_space_fig.rotate_sel_figure(axis='z', angle=angle)
    patched_fig = Patch()
    patched_fig['data'][-1] = content_vars.active_space_fig.fig.data[-1]
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('reset_active_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def reset_active(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.reset_active()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig['data']
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('inverse_active_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def inverse_active(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.invert_active()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig['data']
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('unknwn_to_active_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def unknown_to_active(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.unknown_to_active()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig['data']
    return patched_fig


@callback(
    Output('rec_space_viewer_graph', 'figure', allow_duplicate=True),
    Input('sel_points_button', 'n_clicks'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def active_to_selected(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    content_vars.active_space_fig.active_to_selected()
    patched_fig = Patch()
    patched_fig['data'] = content_vars.active_space_fig.fig['data']
    return patched_fig


@callback(
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Output('map_input_container', 'children'),
    Input('dropdown_map_switcher', 'value'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def map_switch(value):
    if value not in ('3d map', '2d map', '1d map'):
        raise dash.exceptions.PreventUpdate()
    real_axes = content_vars.real_axes
    if real_axes is None:
        raise dash.exceptions.PreventUpdate()

    if value == '3d map':
        if real_axes.count('true') < 3:
            return list((True, diff_map3d_axes_error.header, diff_map3d_axes_error.body)), list()
        else:
            input_axes_table = html.Div(dash_table.DataTable(
                data=pd.DataFrame({'x_axis': '', 'y_axis': '', 'z_axis': ''}, index=[0]).to_dict('records'),
                columns=[
                    {'id': 'x_axis', 'name': 'x', 'type': 'numeric', 'presentation': 'dropdown'},
                    {'id': 'y_axis', 'name': 'y', 'type': 'numeric', 'presentation': 'dropdown'},
                    {'id': 'z_axis', 'name': 'z', 'type': 'numeric', 'presentation': 'dropdown'},
                ],
                editable=True,
                style_cell={
                    'height': '0px',
                    'width': 60,
                },
                fill_width=False,
                dropdown={
                    'x_axis': {
                        'options': [{'label': f'{name}', 'value': no} for
                                    no, name in enumerate(exp1.axes_names) if real_axes[no] == 'true']},
                    'y_axis': {
                        'options': [{'label': f'{name}', 'value': no + 1} for
                                    no, name in enumerate(exp1.axes_names[1:]) if real_axes[1:][no] == 'true']},
                    'z_axis': {
                        'options': [{'label': f'{name}', 'value': no} for
                                    no, name in enumerate(exp1.axes_names) if real_axes[no] == 'true']},
                },
                id='map3d_axes_table'
            ),
                style={'display': 'inline-block'}
            )
            input_angles_table = apg.generate_angle_table(exp_inst=exp1, id_='map3d_angles_table',
                                                          style={'display': 'inline-block',
                                                                 'margin-left': '3vw', },
                                                          style_cell={'height': '0px', 'width': 60, })

            input_range_step_table = (
                html.Div(
                    dash_table.DataTable(
                        data=pd.DataFrame(
                            {'x_min': 0, 'x_max': 360, 'x_step': 4, 'z_min': 0, 'z_max': 360, 'z_step': 4},
                            index=[0]).to_dict(
                            'records'),
                        columns=[
                            {'id': 'x_min', 'name': 'x min', 'type': 'numeric'},
                            {'id': 'x_max', 'name': 'x max', 'type': 'numeric'},
                            {'id': 'x_step', 'name': 'x step', 'type': 'numeric'},
                            {'id': 'z_min', 'name': 'z min', 'type': 'numeric'},
                            {'id': 'z_max', 'name': 'z max', 'type': 'numeric'},
                            {'id': 'z_step', 'name': 'z step', 'type': 'numeric'},

                        ],
                        editable=True,
                        style_cell={
                            'height': '0px',
                            'width': 60,
                        },
                        fill_width=False,
                        id='map3d_range_step_table',

                    ),
                    style={'display': 'inline-block',
                           'margin-left': '3vw', }
                )
            )
            return list((False, None, None)), html.Div(
                children=[input_axes_table, input_angles_table, input_range_step_table])
    elif value == '2d map':
        if real_axes.count('true') < 2:
            return list((True, diff_map2d_axes_error.header, diff_map2d_axes_error.body)), list()
        else:
            input_axes_table = html.Div(dash_table.DataTable(
                data=pd.DataFrame({'x_axis': '', 'y_axis': ''}, index=[0]).to_dict('records'),
                columns=[
                    {'id': 'x_axis', 'name': 'x', 'type': 'numeric', 'presentation': 'dropdown'},
                    {'id': 'y_axis', 'name': 'y', 'type': 'numeric', 'presentation': 'dropdown'},
                ],
                editable=True,
                style_cell={
                    'height': '0px',
                    'width': 60,
                },
                fill_width=False,
                dropdown={
                    'x_axis': {
                        'options': [{'label': f'{name}', 'value': no} for
                                    no, name in enumerate(exp1.axes_names) if real_axes[no] == 'true']},
                    'y_axis': {
                        'options': [{'label': f'{name}', 'value': no + 1} for
                                    no, name in enumerate(exp1.axes_names[1:]) if real_axes[1:][no] == 'true']},
                },
                id='map2d_axes_table'
            ),
                style={'display': 'inline-block'}
            )
            input_angles_table = apg.generate_angle_table(exp_inst=exp1, id_='map2d_angles_table',
                                                          style={'display': 'inline-block',
                                                                 'margin-left': '3vw', },
                                                          style_cell={'height': '0px', 'width': 60, })

            input_range_step_table = (
                html.Div(
                    dash_table.DataTable(
                        data=pd.DataFrame(
                            {'x_min': 0, 'x_max': 360, 'x_step': 1},
                            index=[0]).to_dict(
                            'records'),
                        columns=[
                            {'id': 'x_min', 'name': 'x min', 'type': 'numeric'},
                            {'id': 'x_max', 'name': 'x max', 'type': 'numeric'},
                            {'id': 'x_step', 'name': 'x step', 'type': 'numeric'},
                        ],
                        editable=True,
                        style_cell={
                            'height': '0px',
                            'width': 60,
                        },
                        fill_width=False,
                        id='map2d_range_step_table',

                    ),
                    style={'display': 'inline-block',
                           'margin-left': '3vw', }
                )
            )
            return list((False, None, None)), html.Div(
                children=[input_axes_table, input_angles_table, input_range_step_table])

    elif value == '1d map':
        if real_axes.count('true') < 1:
            return list((True, diff_map1d_axes_error.header, diff_map1d_axes_error.body)), list()
        else:

            input_axes_table = html.Div(dash_table.DataTable(
                data=pd.DataFrame({'y_axis': ''}, index=[0]).to_dict('records'),
                columns=[
                    {'id': 'y_axis', 'name': 'y', 'type': 'numeric', 'presentation': 'dropdown'},
                ],
                editable=True,
                style_cell={
                    'height': '0px',
                    'width': 60,
                },
                fill_width=False,
                dropdown={
                    'y_axis': {
                        'options': [{'label': f'{name}', 'value': no} for
                                    no, name in enumerate(exp1.axes_names) if real_axes[no] == 'true']},

                },
                id='map1d_axes_table'
            ),
                style={'display': 'inline-block'}
            )
            input_angles_table = apg.generate_angle_table(exp_inst=exp1, id_='map1d_angles_table',
                                                          style={'display': 'inline-block',
                                                                 'margin-left': '3vw', },
                                                          style_cell={'height': '0px', 'width': 60, })

            input_range_table = (
                html.Div(
                    dash_table.DataTable(
                        data=pd.DataFrame(
                            {'x_min': 0, 'x_max': 360},
                            index=[0]).to_dict(
                            'records'),
                        columns=[
                            {'id': 'x_min', 'name': 'x min', 'type': 'numeric'},
                            {'id': 'x_max', 'name': 'x max', 'type': 'numeric'},
                        ],
                        editable=True,
                        style_cell={
                            'height': '0px',
                            'width': 60,
                        },
                        fill_width=False,
                        id='map1d_range_table',

                    ),
                    style={'display': 'inline-block',
                           'margin-left': '3vw', }
                )
            )
            return list((False, None, None)), html.Div(
                children=[input_axes_table, input_angles_table, input_range_table])


@callback(
    Output('diffraction_map_graph', 'figure'),
    Output('diff_map_workaround_P', 'children'),
    Input('map_selected_button', 'n_clicks'),
    State('dropdown_map_switcher', 'value'),
    State('map_input_container', 'children'),
    State('diff_map_detector', 'rowData')
)
@mylogger(level='DEBUG')
def calculate_diff_map(n_clicks, map_type, data_container, det_data):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    if content_vars.active_space_fig is None:
        raise dash.exceptions.PreventUpdate()
    det_data = det_data[0]
    selected_trace_n = content_vars.active_space_fig.n_traces - 1
    selected_reflections = content_vars.active_space_fig.fig.data[selected_trace_n]['customdata'][:, :].copy()
    selected_reflections.flags.writeable = True

    axes_data = data_container['props']['children'][0]['props']['children']['props']['data'][0]
    angles_data = data_container['props']['children'][1]['props']['children']['props']['data'][0]
    range_step_data = data_container['props']['children'][2]['props']['children']['props']['data'][0]
    if map_type == '3d map':
        N_OF_RAND_REF = 20
        selected_n = selected_reflections.shape[0]
        if selected_n <= N_OF_RAND_REF:
            reflections = selected_reflections[:, :3]
        else:
            np.random.shuffle(selected_reflections)
            reflections = selected_reflections[: N_OF_RAND_REF, :3]
        yxz_axes = (axes_data['y_axis'], axes_data['x_axis'], axes_data['z_axis'])
        angles = list(angles_data.values())
        range_x, step_x = [(range_step_data['x_min'], range_step_data['x_max']), range_step_data['x_step']]
        range_z, step_z = [(range_step_data['z_min'], range_step_data['z_max']), range_step_data['z_step']]
        if det_data['factor_detector']:
            det_prm = {'dist': det_data['d_dist'],
                       'rot': (det_data['rot_x'], det_data['rot_y'], det_data['rot_z']),
                       'orientation': det_data['orientation'],
                       'disp_y': det_data['disp_y'],
                       'disp_z': det_data['disp_z']}
            fig = exp1.generate_diffraction_map_3d(reflections=reflections, yxz_rotations=yxz_axes,
                                                   xz_steps=(step_x, step_z), xz_ranges=(range_x, range_z),
                                                   factor_detector=True, det_prm=det_prm, initial_angles=angles,
                                                   check_collisions=det_data['factor_collision'],
                                                   factor_obstacles=det_data['factor_obstacles'])
        else:
            fig = exp1.generate_diffraction_map_3d(reflections=reflections, yxz_rotations=yxz_axes,
                                                   xz_steps=(step_x, step_z), xz_ranges=(range_x, range_z),
                                                   initial_angles=angles, check_collisions=det_data['factor_collision'],
                                                   factor_obstacles=det_data['factor_obstacles'])
        return fig, '3d map'


    elif map_type == '2d map':
        N_OF_RAND_REF = 100
        selected_n = selected_reflections.shape[0]
        if selected_n <= N_OF_RAND_REF:
            reflections = selected_reflections[:, :3]
        else:
            np.random.shuffle(selected_reflections)
            reflections = selected_reflections[: N_OF_RAND_REF, :3]

        yx_axes = (axes_data['y_axis'], axes_data['x_axis'])
        angles = list(angles_data.values())
        range_x, step_x = [(range_step_data['x_min'], range_step_data['x_max']), range_step_data['x_step']]
        if det_data['factor_detector']:
            det_prm = {'dist': det_data['d_dist'],
                       'rot': (det_data['rot_x'], det_data['rot_y'], det_data['rot_z']),
                       'orientation': det_data['orientation'],
                       'disp_y': det_data['disp_y'],
                       'disp_z': det_data['disp_z']}
            fig = exp1.generate_diffraction_map_2d(reflections=reflections, yx_rotations=yx_axes,
                                                   x_step=step_x, x_range=range_x,
                                                   factor_detector=True, det_prm=det_prm, initial_angles=angles,
                                                   check_collisions=det_data['factor_collision'],
                                                   factor_obstacles=det_data['factor_obstacles'])
        else:
            fig = exp1.generate_diffraction_map_2d(reflections=reflections, yx_rotations=yx_axes,
                                                   x_step=step_x, x_range=range_x,
                                                   factor_detector=False, initial_angles=angles,
                                                   check_collisions=det_data['factor_collision'],
                                                   factor_obstacles=det_data['factor_obstacles'])
        return fig, '2d map'

    elif map_type == '1d map':
        reflections = selected_reflections[:, :3]
        hkl_orig = selected_reflections[:, 3:]

        y_axis = axes_data['y_axis']
        angles = list(angles_data.values())
        if det_data['factor_detector']:
            det_prm = {'dist': det_data['d_dist'],
                       'rot': (det_data['rot_x'], det_data['rot_y'], det_data['rot_z']),
                       'orientation': det_data['orientation'],
                       'disp_y': det_data['disp_y'],
                       'disp_z': det_data['disp_z']}
            fig = exp1.generate_diffraction_map_1d(reflections=reflections, original_hkl=hkl_orig, rotation=y_axis,
                                                   initial_angles=angles, det_prm=det_prm, factor_detector=True,
                                                   check_collisions=det_data['factor_collision'],
                                                   factor_obstacles=det_data['factor_obstacles'])
        else:
            fig = exp1.generate_diffraction_map_1d(reflections=reflections, original_hkl=hkl_orig, rotation=y_axis,
                                                   initial_angles=angles, check_collisions=det_data['factor_collision'],
                                                   factor_obstacles=det_data['factor_obstacles'])

        return fig, '1d map'


@mylogger(level='DEBUG')
def generate_choose_scan_dropdown(id_, style, style_cell):
    if exp1.axes_rotations is None:
        raise dash.exceptions.PreventUpdate()
    real_axes = exp1.axes_real
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
                            no, name in enumerate(exp1.axes_names) if real_axes[no] == 'true']}, },
        id=f'{id_}'
    ),
        style=style
    )
    return input_axes_table


@mylogger(level='DEBUG')
def generate_angle_table(id_, style, style_cell):
    real_axes = content_vars.real_axes
    input_angles_table = html.Div(dash_table.DataTable(
        data=pd.DataFrame(
            {f'{name}_{no}': content_vars.axes_angles[no] for no, name in enumerate(exp1.axes_names)},
            index=[0]).to_dict('records'),
        columns=[{'id': f'{name}_{no}', 'name': f'{name}', 'type': 'numeric'} if real_axes[no] == 'true' else {
            'id': f'{name}_{no}', 'name': f'{name}', 'type': 'numeric', 'editable': False} for no, name in
                 enumerate(exp1.axes_names)
                 ],
        editable=True,
        style_cell=style_cell,
        fill_width=False,
        id=id_,
    ),
        style=style
    )
    return input_angles_table


@callback(
    Output('ref_at_selected_range', 'children'),
    Output('u_ref_at_selected_range', 'children'),
    Output('o_ref_at_selected_range', 'children'),
    Input('diffraction_map_graph', 'relayoutData'),
    State('diff_map_workaround_P', 'children'),
    State('map_input_container', 'children'),
    State('diffraction_map_graph', 'figure'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def calc_1d_section(relayoutdata, map_type, data_container, fig):
    if map_type != '1d map':
        raise dash.exceptions.PreventUpdate()
    keys = list(relayoutdata.keys())
    if 'dragmode' in keys:
        raise dash.exceptions.PreventUpdate()

    if 'xaxis.autorange' in keys:
        start_angle = 0
        end_angle = 360
    else:
        xaxis_r0, xaxis_r1 = relayoutdata['xaxis.range[0]'], relayoutdata['xaxis.range[1]']
        start_angle = xaxis_r0 if xaxis_r0 <= xaxis_r1 else xaxis_r1
        end_angle = xaxis_r1 if xaxis_r0 <= xaxis_r1 else xaxis_r0
    start_angle, end_angle = np.deg2rad(start_angle), np.deg2rad(end_angle)

    diff_angles = np.array(fig['data'][0]['x'])
    diff_angles = np.deg2rad(diff_angles)
    reflections_encoded = np.array(fig['data'][0]['customdata'])
    hkl_encoded = reflections_encoded[0]
    hkl_orig_encoded = reflections_encoded[1]
    min_, max_, range_ = Sample.angle_range(start_rad=start_angle, end_rad=end_angle)
    mask = Sample.angles_in_sweep(angles_array=diff_angles, sweep_type=range_, start=min_, end=max_, return_bool=True)
    hkl_encoded = hkl_encoded[mask]
    hkl_orig_encoded = hkl_orig_encoded[mask]

    n_of_all_reflections = len(hkl_encoded)
    n_of_unique_reflections = len(np.unique(hkl_encoded))
    n_of_unique_orig_reflections = len(np.unique(hkl_orig_encoded))

    all_str = f'{n_of_all_reflections} reflections at selected range'
    uniq_str = f'{n_of_unique_reflections} unique reflections at selected range'
    uniq_orig_str = f'{n_of_unique_orig_reflections} independent reflections at selected range'
    return all_str, uniq_str, uniq_orig_str


@callback(
    Output('completeness_dag', 'rowData', allow_duplicate=True),
    Output('stored_cumulative_dag_data', 'data'),
    Output('hidden_div_4', 'children'),
    Input('update_cumulative_data_btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_cumulative_data(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    try:
        exp1.refresh_cdcc()
    except CDCCError as e:
        return no_upd, no_upd, (True, e.error_modal_content.header, e.error_modal_content.body)
    rowdata = sf.generate_rowdata_cumulative_ag(exp1)
    return rowdata, rowdata, no_upd


@callback(
    Output("completeness_dag", "rowData", allow_duplicate=True),
    Input("completeness_dag", "virtualRowData"),
    State("completeness_dag", "rowData"),
    prevent_initial_call=True
)
def update_row_order(virtual_data, current_data):
    if not virtual_data or not current_data:
        return no_upd

    order_mapping = {row['id']: row for row in current_data}

    updated_data = []
    for i, virt_row in enumerate(virtual_data):
        original_row = order_mapping.get(virt_row['id'])
        if original_row:
            new_row = {
                **original_row,
                "order": i
            }
            updated_data.append(new_row)
    return updated_data


@callback(
    Output('cumulative_plot_graph', 'figure', allow_duplicate=True),
    Output('stored_cumulative_plot', 'data', allow_duplicate=True),
    Output("completeness_dag", "rowData", allow_duplicate=True),
    Output("stored_cumulative_dag_data", "data", allow_duplicate=True),
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Input("calc_cumulative_reorder_comp_plot_btn", "n_clicks"),
    State("completeness_dag", "rowData"),
    State("cumulative_range", "rowData"),
    prevent_initial_call=True
)
def sort_run_calc_comp_plots(n_clicks, dag_data, d_range):
    if n_clicks == 0 or not dag_data:
        raise dash.exceptions.PreventUpdate()

    try:
        input_params = sf.cumulative_dag_row_data_processing(dag_data)
    except CalcCumulativeCompletenessError as e:
        return no_upd, no_upd, no_upd, no_upd, (True, e.error_modal_content.header, e.error_modal_content.body)

    sf.process_d_range(exp1, d_range)
    sf.update_data_container(exp1, input_params, len(dag_data))

    fig, completeness = exp1.generate_1d_comp_cumulative_plot(order=True)
    updated_dag = sf.update_completeness_data(dag_data, completeness)

    return fig, fig, updated_dag, updated_dag, no_upd


@callback(
    Output('cumulative_plot_graph', 'figure', allow_duplicate=True),
    Output('stored_cumulative_plot', 'data', allow_duplicate=True),
    Output("completeness_dag", "rowData", allow_duplicate=True),
    Output("stored_cumulative_dag_data", "data", allow_duplicate=True),
    Output('hidden_div_4', 'children', allow_duplicate=True),
    Input("calc_cumulative_ordered_comp_plot_btn", "n_clicks"),
    State("completeness_dag", "rowData"),
    State("cumulative_range", "rowData"),
    prevent_initial_call=True
)
def run_calc_comp_plots(n_clicks, dag_data, d_range):
    if n_clicks == 0 or not dag_data:
        raise dash.exceptions.PreventUpdate()

    try:
        input_params = sf.cumulative_dag_row_data_processing(dag_data)
    except CalcCumulativeCompletenessError as e:
        return no_upd, no_upd, no_upd, no_upd, (True, e.error_modal_content.header, e.error_modal_content.body)

    sf.process_d_range(exp1, d_range)
    sf.update_data_container(exp1, input_params, len(dag_data))
    fig, completeness = exp1.generate_1d_comp_cumulative_plot(
        order=False,
        permutation_indices=input_params['run_order']
    )
    updated_dag = sf.update_completeness_data(dag_data, completeness)

    return fig, fig, updated_dag, updated_dag, dash.no_update


app.clientside_callback(
    """
    async function(cellValueChanges) {
        if (!cellValueChanges || cellValueChanges.length === 0) {
            return window.dash_clientside.no_update;
        }

        try {
            const gridApi = await dash_ag_grid.getApiAsync("diff_map_detector");

            if (!gridApi || typeof gridApi.setColumnsVisible !== 'function') {
                console.warn('Grid API not available');
                return window.dash_clientside.no_update;
            }

            for (const change of cellValueChanges) {
                if (change.colId === 'orientation') {
                    const newVisibility = change.value === 'independent';

                    if (!gridApi.getModel() || !gridApi.getColumnDef('disp_y')) {
                        console.warn('Grid not ready, retrying...');
                        setTimeout(() => {
                            gridApi.setColumnsVisible(['disp_y','disp_z'], newVisibility);
                        }, 300);
                        return window.dash_clientside.no_update;
                    }

                    try {
                        gridApi.setColumnsVisible(['disp_y','disp_z'], newVisibility);
                    } catch (innerError) {
                        console.error('First try failed:', innerError);
                        setTimeout(() => {
                            try {
                                gridApi.setColumnsVisible(['disp_y','disp_z'], newVisibility);
                            } catch (finalError) {
                                console.error('Final error:', finalError);
                            }
                        }, 300);
                    }
                }
            }
        } catch (error) {
            console.error('Callback error:', error);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('diff_map_detector', 'id'),
    Input('diff_map_detector', 'cellValueChanged')
)


@callback(
    Output({'type': 'obstacle_table', 'index': MATCH}, 'hidden_columns'),
    Input({'type': 'obstacle_table', 'index': ALL}, 'data'),
    prevent_initial_call=True
)
def obst_table_hidden_control(data_):
    data_changed = ctx.triggered[0]['value'][0]
    table_index = sf.find_index_digit(tuple(data_changed.keys())[0])
    hidden_columns = sf.prepare_hidden_columns(table_index, data_changed)
    return hidden_columns


@callback(
    Output({'type': 'linked_obstacle_table', 'index': MATCH}, 'hidden_columns'),
    Input({'type': 'linked_obstacle_table', 'index': ALL}, 'data'),
    prevent_initial_call=True
)
def obst_table_hidden_control(data_):
    data_changed = ctx.triggered[0]['value'][0]
    table_index = sf.find_index_digit(tuple(data_changed.keys())[0])
    hidden_columns = sf.prepare_hidden_columns(table_index, data_changed)
    return hidden_columns


@callback(
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


@callback(Output('stored_linked_obstacles_div', 'data', allow_duplicate=True),
          Input('clear_linked_obstacles_button', 'n_clicks'),
          prevent_initial_call=True
          )
@mylogger(level='DEBUG')
def delete_obstacles(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    exp1.clear_linked_obstacles()
    return None


@callback(
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

@callback(Output('download_linked_obstacles', 'data'),
          Input('download_linked_obstacles_btn', 'n_clicks'))
@mylogger(level='DEBUG')
def download_linked_obstacles(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    data_json = exp1.json_export('linked_obstacles')
    return dict(content=data_json, filename='linked_obstacles_input.json')

@callback(
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
def load_obstacles(json_f, table_num):
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

if __name__ == '__main__':
    app.run(debug=True, )
