from dash import html, dcc, Input, Output, State, no_update as no_upd
import dash_bootstrap_components as dbc


from app import app, server
from global_state import content_vars


from pages import home
from pages import instrument
from pages import strategy
from pages import calculation


from logger.my_logger import mylogger


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
    style=SIDEBAR_STYLE
)

content = html.Div(id="page-content", style=CONTENT_STYLE)


STORE_TYPE = 'memory'


app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,



    dcc.Store(data=None, id='old_url', storage_type=STORE_TYPE),
    dcc.Store(data=False, id='rec_space_viewer_graph_has_data', storage_type=STORE_TYPE),


    dcc.Store(data=False, id='home_page_stored_flag', storage_type=STORE_TYPE),
    dcc.Store(data=None, id='stored_UB_table', storage_type=STORE_TYPE),
    dcc.Store(data=None, id='stored_parameters_table', storage_type=STORE_TYPE),


    dcc.Store(data=list(), id='stored_log_collision_check', storage_type=STORE_TYPE),
    dcc.Store(data=False, id='page-1_stored_flag', storage_type=STORE_TYPE),
    dcc.Store(data=None, id='stored_wavelength_val', storage_type=STORE_TYPE),
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


    dcc.Store(data=False, id='page-2_stored_flag', storage_type=STORE_TYPE),
    dcc.Store(data=None, id='stored_runs_div', storage_type=STORE_TYPE),
    dcc.Store(data=0, id='stored_runs_num', storage_type=STORE_TYPE),


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




def get_rec_figure(old_url):
    """
    Вспомогательная функция для сохранения состояния графика обратного пространства
    при уходе со страницы расчетов.
    """
    if old_url != '/page-3' or old_url is None or content_vars.active_space_fig is None:
        return no_upd
    else:
        return content_vars.active_space_fig.fig


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

        return home.layout, True, no_upd, no_upd, no_upd, pathname, rec_fig
    elif pathname == "/page-1":

        return instrument.layout, no_upd, True, no_upd, no_upd, pathname, rec_fig
    elif pathname == "/page-2":

        return strategy.layout, no_upd, no_upd, True, no_upd, pathname, rec_fig
    elif pathname == "/page-3":

        return calculation.layout, no_upd, no_upd, no_upd, True, pathname, rec_fig


    return html.Div(
        [
            html.H2("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    ), no_upd, no_upd, no_upd, no_upd, pathname, rec_fig


if __name__ == '__main__':
    app.run(debug=True)