
import datetime
from datetime import date
import pandas as pd
from dash import html, dcc, callback, Dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from utils import *


try:
    tickers = get_tickers()
except Exception as e:
    print(e)

optim_portfolio = None

app = Dash(__name__, use_pages=False, external_stylesheets=[dbc.themes.LUX])

server = app.server
app.layout = html.Div([
    html.H1("Portfolio analyzer for S&P500 stocks"),
    html.P("1. Select two or more stocks, time span, and visualize returns and volatilites."),
    html.P("2. Press Optimize portfolio weights to get weights for maximum Sharpe ratio"),
    html.P("3. Select number of days and percentage, and press Simulate Value at Risk to get VAR values for the optimized portfolio"),
    dbc.Label("Select at least two stocks from the dropdown menu", html_for="dropdown"),
    html.Div([
        dcc.Dropdown(id="ticker_dropdown", options=tickers.Security.values, value="", multi=True)
    ],
    style={"width": "30%", "text-align":"center"}),
    dbc.Label("Select a time frame of at least 15 days for calculating average daily returns and volatility (risk)", html_for="datepicker"),
    html.Br(),
    dcc.DatePickerRange(
        id="datepicker",
        min_date_allowed=date(2000, 1, 1),
        max_date_allowed=date.today(),
        start_date_placeholder_text="Start Period",
        end_date_placeholder_text="End Period",
        initial_visible_month=date.today(),
    ),
    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Button("Plot stock data", id="plot_data", n_clicks=0, size="lg", className="me-1")),
        dbc.Col(dcc.Loading(
                id = "loading-price",
                type = "default",
                children=[html.Div(id="stock_prices", style={"display": "inline-block", "width": "50%"}),
                        html.Div(id="returns_volas", style={"display": "inline-block", "width": "50%"})]
                ),md=9
            )
        ] 
        ),

    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Button("Optimize portfolio weights", id="optimize_portfolio", size="lg", className="me-1")),
        dbc.Col(dcc.Loading(
                id = "loading-portfolio",
                type = "default",
                children=[html.Div(id="optimizer", style={"display": "inline-block", "width": "80%", "text-align":"center"})]
                ),md=9
            )
        ] 
        ),

    html.Br(),
    html.P("Select number of days to simulate Value at risk"),
    html.Div([dcc.Dropdown(id="days_risk", options=[10, 50, 100], value="", multi=False)],style={"width": "30%", "text-align":"center"}),
    html.Br(),
    html.P("Select percentage for Value at Risk"),
    html.Div([dcc.Dropdown(id="percentage_risk", options=[5, 10, 15], value="", multi=False)],style={"width": "30%", "text-align":"center"}),
    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Button("Simulate value at risk", id="simulate_risk", size="lg", className="me-1")),
        dbc.Col(dcc.Loading(
                id = "loading-risk",
                type = "default",
                children=[html.Div(id="simulator", style={"display": "inline-block", "width": "80%", "text-align": "center"})]
                ), md=9
            )
        ]
        ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br()
])

@callback(
    Output("stock_prices", "children"),
    Output("returns_volas", "children"),
    Input("plot_data", "n_clicks"),
    [State("ticker_dropdown", "value"),
     State("datepicker", "start_date"),
     State("datepicker", "end_date")]
)

def plot_data(n_clicks, selected_stocks, start_date, end_date):
    if not start_date or not end_date or len(selected_stocks) < 2 or (datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.datetime.strptime(start_date, "%Y-%m-%d")).days < 15:
        raise PreventUpdate
    symbols = tickers[tickers["Security"].isin(selected_stocks)]["Symbol"]
    port = Portfolio(list(symbols), start_date, end_date)
    port.download_close_data()
    port.get_daily_returns()
    port.get_avg_daily_returns()
    port.get_volatility()
    fig1 = go.Figure()
    for sym in symbols:
        fig1.add_trace(go.Scatter(mode="markers", x=[port.volatility[sym]], y=[port.avg_daily_returns[sym]*100], name=tickers[tickers["Symbol"] == sym]["Security"].values[0], 
                                  marker=dict(size=15, line=dict(width=2))
                                  )
                                  )   
    fig1.update_layout(xaxis_title="Volatility", yaxis_title="Average of daily returns")

    fig2 = go.Figure()
    for sym in symbols:
       fig2.add_trace(go.Line(x=port.close_data.index.values, y=port.close_data[sym].values, name=tickers[tickers["Symbol"] == sym]["Security"].values[0]))

    fig2.update_layout(xaxis_title="Date", yaxis_title="Closing price $")

    return dcc.Graph(figure=fig2), dcc.Graph(figure=fig1)

@callback(
    Output("optimizer", "children"),
    Input("optimize_portfolio", "n_clicks"),
    [State("ticker_dropdown", "value"),
     State("datepicker", "start_date"),
     State("datepicker", "end_date")]
)

def optimise_portfolio(n_clicks, selected_stocks, start_date, end_date):
    if not start_date or not end_date or len(selected_stocks) < 2 or (datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.datetime.strptime(start_date, "%Y-%m-%d")).days < 15:
        raise PreventUpdate
    symbols = tickers[tickers["Security"].isin(selected_stocks)]["Symbol"]
    port = Portfolio(list(symbols), start_date, end_date)
    port.download_close_data()
    port.get_daily_returns()
    port.get_avg_daily_returns()
    port.get_volatility()
    port.optimize()
    port.get_optimization_results()
    global optim_portfolio
    optim_portfolio = port
    names = []
    for sym in port.series.index:
        names.append(tickers[tickers["Symbol"] == sym]["Security"].values[0])
    if port.results.success:
        fig = go.Figure([go.Bar(x=names, y=port.series.values)])
        fig.update_layout(xaxis_title="Stock symbol", yaxis_title="Weight")
        return [f"Optimal allocation found with Sharpe ratio of {np.round(port.optim_sharpe, decimals=3)}. Expected return for the portfolio is {np.round(port.optim_returns, decimals=3)}% with volatility of {np.round(port.optim_vola, decimals=3)}", dcc.Graph(figure=fig)]
    else:
        return "Optimal allocation not found. Select other stocks and/or timeframe"

@callback(
    Output("simulator", "children"),
    Input("simulate_risk", "n_clicks"),
    [State("days_risk", "value"),
     State("percentage_risk", "value")]
)

def simulate_risk(n_clicks, days, percentage):
    if optim_portfolio == None or days == "" or percentage == "":
        raise PreventUpdate
    optim = mc_simulator(optim_portfolio, days)
    var = np.abs(np.percentile(optim.iloc[-1,:].values, percentage) - 1)
    fig = go.Figure()
    for path in optim.columns:
        fig.add_trace(go.Line(x=optim[path].index, y=optim[path], line=dict(color="rgba(0, 0, 255, 0.2)")))
    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis_title="Day", yaxis_title="Portfolio value")
    return [f"{optim.shape[0]}-day VAR for {percentage}% is {np.round(var*100, 2)}% of portfolio value", dcc.Graph(figure=fig)]

if __name__ == '__main__':
    app.run(debug=True)