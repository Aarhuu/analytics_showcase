from datetime import date

import dash
from dash import html, dcc, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from utils import *

dash.register_page(__name__)

strategies = ["Fixed low/high band", "Moving average"]

layout = html.Div([
    html.H1("Simulation of battery storage system"),
    html.P("The simulation presents a simple low/high limit strategy for operating \
           a battery and local solar panel system."),
    html.P("Used data is from February 2024 in Finland"),
    html.P("Fill the parameters and press 'Run' to visualize results."),
    dcc.Input(id="panel_area", type="number", placeholder="Panel area in square meters"),
    html.Br(),
    dcc.Input(id="avg_energy_use", type="number", placeholder="Average energy use in kWh"),
    html.Br(),
    dcc.Input(id="storage_capacity", type="number", placeholder="Battery capacity in kWh"),
    html.Br(),
    html.Div([  
        dcc.Dropdown(id="optimizer_strategy", options=strategies, value="")
        ],
        style={"width": "30%"}
    )
])