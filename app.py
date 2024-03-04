
from datetime import date

import dash
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

app = Dash(__name__, use_pages=False, external_stylesheets=[dbc.themes.LUX])

server = app.server
app.layout = html.Div([
    html.H1("Portfolio sharpe ratio optimizer for S&P500 stocks"),
    dbc.Label("Select at least two stocks from the dropdown menu", html_for="dropdown"),
    html.Div([
        dcc.Dropdown(id="ticker_dropdown", options=tickers.Security.values, value="", multi=True)
    ],
    style={"width": "30%"}),
    dbc.Label("Select a time frame for calculating average daily returns and volatility (risk)", html_for="datepicker"),
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
    html.Div([
    dbc.Button("Plot prices, average daily returns and volatility", id="plot_data", n_clicks=0),
    ],
    className="me-1 mt-1 btn btn-primary", style={"width": "40vh"}),
    html.Br(),
    html.Div([
    dbc.Button("Optimize portfolio weights", id="optimize_portfolio", n_clicks=0),
    ],
    className="me-1 mt-1 btn btn-primary", style={"width": "40vh"}),
    html.Div([
        html.Div(id="stock_prices", style={"display": "inline-block", "width": "50%"}),
        html.Div(id="returns_volas", style={"display": "inline-block", "width": "50%"})
    ]),
    html.Div([
        html.Div(id="optimizer", style={"display": "inline-block", "width": "80%"})
    ])
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
    if not start_date or not end_date or len(selected_stocks) < 2:
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
    if not start_date or not end_date or len(selected_stocks) < 2:
        raise PreventUpdate
    symbols = tickers[tickers["Security"].isin(selected_stocks)]["Symbol"]
    port = Portfolio(list(symbols), start_date, end_date)
    port.download_close_data()
    port.get_daily_returns()
    port.get_avg_daily_returns()
    port.get_volatility()
    port.optimize()
    port.get_optimization_results()
    names = []
    for sym in port.series.index:
        names.append(tickers[tickers["Symbol"] == sym]["Security"].values[0])
    if port.results.success:
        fig = go.Figure([go.Bar(x=names, y=port.series.values)])
        fig.update_layout(xaxis_title="Stock symbol", yaxis_title="Weight")
        return [f"Optimal allocation found with Sharpe ratio of {np.round(port.optim_sharpe, decimals=3)}. Extrapolated annual return for the portfolio is {np.round(port.optim_returns, decimals=3)}% with volatility of {np.round(port.optim_vola, decimals=3)}", dcc.Graph(figure=fig)]
    else:
        return "Optimal allocation not found. Select other stocks and/or timeframe"


if __name__ == '__main__':
    app.run(debug=False)