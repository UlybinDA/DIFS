from dash import html, dcc, dash_table, Input, Output, State, Patch, ALL, ctx, no_update as no_upd
from dash.dash_table.Format import Format, Scheme
from dash_extensions import Keyboard
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import copy
import dash
from dash import MATCH, ALL
from app import app
from global_state import content_vars, exp1, point_groups, centrings
import services.service_functions as sf
from services.exceptions.exceptions import *
from assets.modals_content import *
import assets.app_gens as apg
from logger.my_logger import mylogger
from sample.sample import Sample

STYLE_BTN_ACTIVE = {'background-color': '#28a745', 'color': 'white', 'margin-right': '5px',
                    'font-size': '12px'}  # Зеленый
STYLE_BTN_INACTIVE = {'background-color': '#dc3545', 'color': 'white', 'margin-right': '5px', 'font-size': '12px',
                      'opacity': '0.6'}  # Красный/Тусклый



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

layout = html.Div([

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
            html.H4('Scan Optimization',
                    style={'margin-top': '50px', 'border-top': '2px solid #ccc', 'padding-top': '10px'}),

            html.Div([
                html.Div([
                    html.Button('Start Optimization', id='start_optimization_btn', n_clicks=0,
                                style={'margin-right': '20px', 'height': '40px'}),
                ], style={'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.Div([
                        html.Label("Target Scan:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                        html.Div(id='opt_scans_buttons_container', style={'display': 'inline-block'}),
                    ], style={'margin-bottom': '10px'}),

                    html.Div([
                        html.Label("Include in Base:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                        html.Div(id='opt_base_runs_container', style={'display': 'inline-block'}),
                    ])
                ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'})

            ], style={'padding': '20px', 'background-color': '#f9f9f9', 'border-bottom': '1px solid #ddd'}),

            html.Div([
                html.Div([
                    html.H6("Settings", style={'margin-bottom': '15px'}),

                    html.Label("Metric:"),
                    dcc.RadioItems(
                        options=[
                            {'label': 'Completeness', 'value': 'completeness'},
                            {'label': 'Redundancy', 'value': 'redundancy'},
                            {'label': 'Multiplicity', 'value': 'multiplicity'}
                        ],
                        value='completeness',
                        id='opt_metric_selector',
                        labelStyle={'display': 'block', 'margin-bottom': '5px'}
                    ),
                    html.Hr(),

                    html.Label("Start Angle:"),
                    dcc.Input(id='opt_input_start', type='number', placeholder='Click on graph...',
                              style={'width': '100%', 'margin-bottom': '10px'}),

                    html.Label("Sweep:"),
                    dcc.Input(id='opt_input_sweep', type='number', placeholder='Click on graph...',
                              style={'width': '100%', 'margin-bottom': '15px'}),

                    html.Hr(),

                    html.Button('Apply Parameters', id='opt_apply_btn', n_clicks=0,
                                style={'width': '100%', 'background-color': '#d4edda', 'margin-bottom': '10px',
                                       'height': '40px'}),
                    html.Button('Revert Changes', id='opt_revert_btn', n_clicks=0,
                                style={'width': '100%', 'background-color': '#f8d7da', 'height': '40px'}),

                    html.Div(id='opt_status_msg', style={'margin-top': '10px', 'color': 'gray', 'font-size': '0.9em'})

                ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px',
                          'background-color': '#fff'}),

                html.Div([
                    dcc.Graph(
                        id='optimization_graph',
                        style={'height': '700px', 'width': '100%'},
                        config={'scrollZoom': True, 'displayModeBar': True}
                    )
                ], style={'width': '78%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})

            ], id='optimization_main_area', style={'display': 'none'})

        ], style={'padding-bottom': '100px'})
        ,
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


def if_val_None_return_no_upd_else_return(val):
    if val is None:
        return no_upd
    else:
        return val




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


@app.callback(
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


@app.callback(
    Input('centring_selector', 'value'),
)
@mylogger(level='DEBUG')
def set_centring_val(value):
    if value == '':
        raise dash.exceptions.PreventUpdate()
    exp1.set_centring(centring=value)


@app.callback(
    Input('point_group_selector', 'value'),
)
@mylogger(level='DEBUG')
def set_pg_val(value):
    if value == '':
        raise dash.exceptions.PreventUpdate()
    exp1.set_pg(pg=value)


@app.callback(
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
def calc_experiment(n_clicks, children, pg, centring):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate()
    d_range = children[0]

    if exp1.centring is None:
        return 'error', list((True, calc_exp_centring_error.header, calc_exp_centring_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd

    if exp1.pg is None:
        return 'error', list((True, calc_exp_pg_error.header, calc_exp_pg_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd

    if d_range['min'] > d_range['max']:
        return 'error', list((True, calc_exp_min_max_error.header, calc_exp_min_max_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd
    if exp1.wavelength is None:
        return 'error', list((True, calc_exp_wavelength_error.header, calc_exp_wavelength_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd
    if exp1.scans is None:
        return 'error', list((True, calc_exp_no_scans_error.header, calc_exp_no_scans_error.body)), {
            'background-color': 'red'}, no_upd, no_upd, no_upd, no_upd

    try:
        exp1.calc_experiment(d_range=(d_range['min'],d_range['max']))
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


@app.callback(Output('hidden_div_4', 'children', allow_duplicate=True),
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


@app.callback(
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


# 1. ОБНОВЛЕННЫЙ КОЛБЭК ИНИЦИАЛИЗАЦИИ
@app.callback(
    [Output('opt_scans_buttons_container', 'children'),
     Output('opt_base_runs_container', 'children'),  # <-- Новый Output
     Output('optimization_main_area', 'style'),
     Output('optimization_graph', 'figure', allow_duplicate=True)],
    Input('start_optimization_btn', 'n_clicks'),
    prevent_initial_call=True
)
def initialize_optimization(n_clicks):
    if not exp1.runs_are_set():
        return no_upd, no_upd, no_upd, no_upd

    target_buttons = []
    for i, scan in enumerate(exp1.scans):
        axis_name = exp1.axes_names[scan[4]] if exp1.axes_names else f"Ax{scan[4]}"
        btn = html.Button(
            f"Select {i}",
            id={'type': 'opt_scan_btn', 'index': i},
            n_clicks=0,
            style={'margin-right': '5px', 'font-size': '12px'}
        )
        target_buttons.append(btn)
    exp1.init_optimizer()
    active_runs = exp1.opt_get_base_runs()
    print(active_runs)
    base_buttons = []
    for i, scan in enumerate(exp1.scans):
        is_active = i in active_runs
        style = STYLE_BTN_ACTIVE if is_active else STYLE_BTN_INACTIVE

        btn = html.Button(
            f"Run {i}",  # Текст кнопки
            id={'type': 'opt_toggle_base_btn', 'index': i},  # Другой тип ID!
            n_clicks=0,
            style=style
        )
        base_buttons.append(btn)
    fig = exp1.opt_select_scan(0)

    return target_buttons, base_buttons, {'display': 'block'}, fig


@app.callback(
    [Output('optimization_graph', 'figure', allow_duplicate=True),
     Output('opt_input_start', 'value', allow_duplicate=True),
     Output('opt_input_sweep', 'value', allow_duplicate=True)],
    Input({'type': 'opt_scan_btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def select_scan_dynamic(n_clicks_list):
    # Определяем, какая кнопка была нажата
    if not any(n_clicks_list):
        return no_upd, no_upd, no_upd

    # dash.callback_context (ctx) позволяет узнать, кто вызвал триггер
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return no_upd, no_upd, no_upd

    scan_index = triggered_id['index']

    # Вызываем метод эксперимента
    fig = exp1.opt_select_scan(scan_index)

    # Получаем текущие параметры этого скана, чтобы обновить инпуты
    current_scan = exp1.scans[scan_index]
    axis_idx = current_scan[4]
    start_angle = current_scan[3][axis_idx]
    sweep = current_scan[5]

    return fig, start_angle, sweep


@app.callback(
    [Output('opt_input_start', 'value'),
     Output('opt_input_sweep', 'value')],
    Input('optimization_graph', 'clickData'),
    prevent_initial_call=True
)
def handle_graph_click(clickData):
    if not clickData:
        return no_upd, no_upd

    # Извлекаем координаты клика
    point = clickData['points'][0]
    x = point.get('x')  # Start Angle
    y = point.get('y')  # Sweep

    return x, y


@app.callback(
    [Output('optimization_graph', 'figure'),
     Output('opt_status_msg', 'children')],
    [Input('opt_apply_btn', 'n_clicks'),
     Input('opt_revert_btn', 'n_clicks'),
     Input('opt_metric_selector', 'value')],
    [State('opt_input_start', 'value'),
     State('opt_input_sweep', 'value')],
    prevent_initial_call=True
)
def update_optimization_params(n_apply, n_revert, metric, start_val, sweep_val):
    trigger_id = ctx.triggered_id

    # 1. Если нажали APPLY
    if trigger_id == 'opt_apply_btn':
        if start_val is None or sweep_val is None:
            return no_upd, "Please select coordinates first."

        # Применяем параметры через "Бога"
        fig = exp1.opt_apply_parameters(float(start_val), float(sweep_val))
        return fig, f"Applied: Start {start_val:.1f}, Sweep {sweep_val:.1f}"

    # 2. Если нажали REVERT
    elif trigger_id == 'opt_revert_btn':
        fig = exp1.opt_revert()
        return fig, "Changes reverted."

    # 3. Если сменили МЕТРИКУ
    elif trigger_id == 'opt_metric_selector':
        fig = exp1.opt_select_graph_type(metric)
        return fig, f"Metric changed to {metric}"

    return no_upd, ""

@app.callback(
    [Output('optimization_graph', 'figure', allow_duplicate=True),
     Output({'type': 'opt_toggle_base_btn', 'index': ALL}, 'style')],
    Input({'type': 'opt_toggle_base_btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def toggle_base_runs(n_clicks_list):
    # Оставляем только проверку на пустоту
    if not n_clicks_list or not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate

    clicked_index = triggered['index']

    current_active_runs = exp1.opt_get_base_runs()
    is_currently_active = clicked_index in current_active_runs

    new_state = not is_currently_active

    # Основное действие
    fig = exp1.opt_toggle_base_run(clicked_index, new_state)

    updated_active_runs = exp1.opt_get_base_runs()

    new_styles = []
    # Используем len(exp1.scans), чтобы гарантировать порядок
    for i in range(len(exp1.scans)):
        if i in updated_active_runs:
            new_styles.append(STYLE_BTN_ACTIVE)
        else:
            new_styles.append(STYLE_BTN_INACTIVE)

    return fig, new_styles