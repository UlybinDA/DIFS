from json import JSONDecodeError
from services.exceptions.exceptions import *
from dash import html, dcc, Input, Output, State, ALL, no_update as no_upd
import dash_bootstrap_components as dbc
import dash

from app import app
from global_state import content_vars, exp1
import services.service_functions as sf
from assets.modals_content import *
import assets.app_gens as apg
from logger.my_logger import mylogger

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

layout = html.Div([

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


def if_val_None_return_no_upd_else_return(val):
    if val is None:
        return no_upd
    else:
        return val


@app.callback(
    Output('runs_div', 'children', allow_duplicate=True),
    Output("page-2_stored_flag", "data", allow_duplicate=True),
    Output('stored_runs_num', 'data', allow_duplicate=True),  # Добавили output для счетчика таблиц
    Input("page-2_stored_flag", "data"),
    State('stored_runs_div', 'data'),
    State('stored_runs_num', 'data'),
    prevent_initial_call=True)
def get_stored_page_2_data(flag, stored_runs_div, stored_table_num):
    if not flag:
        raise dash.exceptions.PreventUpdate()
    if exp1.runs_are_set() and exp1.goniometer_is_set():
        try:
            new_children, new_table_num = exp1.restore_runs_tables(table_num=0)

            return new_children, False, new_table_num
        except Exception as e:
            print(f"Error restoring runs from exp1: {e}")
    output_children = if_val_None_return_no_upd_else_return(stored_runs_div)
    return output_children, False, stored_table_num


@app.callback(
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


@app.callback(
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
            (True, 'Add run warning',
             'Before adding scans, enter the instrument model of the goniometer')), children, table_num
    new_children = children.copy()
    real_axes = content_vars.real_axes
    axes_angles = content_vars.axes_angles
    names = exp1.axes_names
    new_div_table = apg.gen_run_table(real_axes, axes_angles, rotations, names, table_num)
    table_num += 1
    new_children += [new_div_table, ]
    return list((False, '', '')), new_children, table_num


@app.callback(
    Output('runs_div', 'children'),
    Input({'type': 'run_delete_div_button', 'index': ALL}, 'n_clicks'),
    State('runs_div', 'children'),
    prevent_initial_call=True
)
@mylogger(level='DEBUG')
def delete_run_row(no, children):
    try:
        index_to_delete = no.index(1)
    except ValueError:
        raise dash.exceptions.PreventUpdate()
    children.pop(index_to_delete)
    return children


@app.callback(
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

    raw_rows = []
    for table_data in data:
        if table_data:
            raw_rows.append(table_data[0])

    try:
        data_dicts = sf.process_runs_to_dicts_list(raw_rows)
    except Exception as e:
        return list((True, "Error", str(e))), {'background-color': 'red'}, no_upd
    if exp1.collision_handler.active:
        try:
            sf.check_collision(exp_inst=exp1, runs=data_dicts)
        except CollisionError as e:
            return (list((True, e.error_modal_content.header, e.error_modal_content.body)), {'background-color': 'red'},
                    no_upd)

    for run in data_dicts:
        try:
            sf.check_run_dict_temp(run)
        except RunsDictError as e:
            return (list((True, e.error_modal_content.header, e.error_modal_content.body)), {'background-color': 'red'},
                    no_upd)
    exp1.scans = list()
    for run in data_dicts:
        exp1.add_scan(**run)
    return list((False, '', '')), {'background-color': 'green'}, children


@app.callback(Output('hidden_div_3', 'children', allow_duplicate=True),
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


@app.callback(
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


@app.callback(
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
