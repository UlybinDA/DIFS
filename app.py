import dash
import dash_bootstrap_components as dbc
import os

css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style_main.css')

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SIMPLEX, css_path],
    suppress_callback_exceptions=True
)
server = app.server