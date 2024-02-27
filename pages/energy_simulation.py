from datetime import date

import dash
from dash import html, dcc, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    dcc.Input(id="grid_cost", type="number", placeholder="Grid transfer cost in €/kWh"),
    html.Br(),
    dcc.Input(id="avg_energy_use", type="number", placeholder="Average energy use in kWh"),
    html.Br(),
    dcc.Input(id="storage_capacity", type="number", placeholder="Battery capacity in kWh"),
    html.Br(),
    html.Div([
        html.P("NOTE: the simulation runs roughly 30 seconds."),
        dbc.Button("Run simulation", id="run_simulation", n_clicks=0)
    ],
    style={"width": "30%"}
    ),
    dcc.Loading(id="simulating_icon", type="default", children=html.Div(id="simulation_results"))
])


@callback(
    Output("simulation_results", "children"),
    Input("run_simulation", "n_clicks"),
    [State("panel_area", "value"),
     State("grid_cost", "value"),
     State("avg_energy_use", "value"),
     State("storage_capacity", "value")]
)

def simulate(n_clicks, panel_area, grid_cost, avg_energy_use, storage_capacity):
    if not panel_area or not grid_cost or not avg_energy_use or not storage_capacity:
        raise PreventUpdate
    energy_system = EnergySystem(panel_area, grid_cost, avg_energy_use, storage_capacity)
    spot_filepath = "Oomi Spot-hintatieto.csv"
    energy_system.get_spot_data(spot_filepath)
    energy_system.simulate_system(step_size=4.5)
    df = pd.DataFrame({"time": energy_system.spot_data["Aika"], 
                       "storage": energy_system.simulated_storage, 
                       "savings": energy_system.simulated_savings, 
                       "h_bands": energy_system.optim_h_bands,
                       "l_bands": energy_system.optim_l_bands})
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    fig1.add_trace(go.Line(x=df["time"], y=energy_system.simulated_storage, name="Storage"), secondary_y=False)
    fig1.add_trace(go.Line(x=df["time"], y=energy_system.simulated_savings, name="Savings"), secondary_y=True)
    fig1.update_yaxes(title_text="Storage (kWh)", secondary_y=False)
    fig1.update_yaxes(title_text="Cash flow (€)", secondary_y=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Line(x=df["time"], y=energy_system.spot_data["c/kWh"], name="SPOT price"))
    fig2.add_trace(go.Line(x=df["time"], y=df["h_bands"], name="High threshold"))
    fig2.add_trace(go.Line(x=df["time"], y=df["l_bands"], name="Low threshold"))
    fig2.update_yaxes(title_text="SPOT price of electricity")
    fig2.update_xaxes(title_text="Date")

    return html.P(f"Optimal result"), dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)