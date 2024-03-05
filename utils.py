from datetime import date

import pandas as pd
import yfinance as yf
import numpy as np
import scipy
from scipy.optimize import minimize, LinearConstraint

def get_tickers():
    return pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

class Portfolio():
    def __init__(self, tickers, start_date, end_date) -> None:
        self.tickers: list[str] = tickers
        self.start_date: date = start_date
        self.end_date: date = end_date
        self.trading_days: float = None
        self.close_data: pd.DataFrame = pd.DataFrame()
        self.daily_returns: pd.Dataframe = pd.DataFrame()
        self.volatility: pd.Series = pd.Series()
        self.avg_daily_returns: pd.Series = pd.Series()
        self.target_return: float = None
        self.results: scipy.OptimizeResult = None
        self.optim_returns: float = None
        self.optim_vola: float = None
        self.stock_paths: pd.DataFrame = pd.DataFrame()

    def download_close_data(self) -> None:
        self.close_data = yf.download(self.tickers, self.start_date, self.end_date).Close
        self.trading_days = self.close_data.shape[0]

    def get_daily_returns(self):
        for _, x in enumerate(self.close_data.columns.values):
            self.daily_returns[x] = self.close_data[x].pct_change().ffill()

    def get_avg_daily_returns(self):
        if self.close_data.shape[0] < 1:
            self.close_data = self.download_close_data()
        if self.daily_returns.shape[0] < 1:
            self.daily_returns = self.get_daily_returns()
        self.avg_daily_returns = self.daily_returns.mean()

    def get_volatility(self) -> None:
        if self.close_data.shape[0] < 1:
            self.close_data = self.download_close_data()
        if self.daily_returns.shape[0] < 1:
            self.daily_returns = self.get_daily_returns()
        self.covariance = self.daily_returns.cov()
        volas = {}
        for _, x in enumerate(self.close_data.columns.values):
            volas[x] = np.sqrt(self.trading_days)*self.daily_returns[x].std()
        self.volatility = pd.Series(data=volas)

    def optimize(self) -> None:

        def portfolio_reverse_returns(w):
            optim = np.dot(w, self.avg_daily_returns)/(np.sqrt(np.dot(w, np.dot(self.covariance, w))*252))
            
            return -optim
        
        weights = [1/len(self.tickers)]*len(self.tickers)
        selections = tuple((0, 1) for i in range(len(self.tickers)))
        constraints = (
        {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
        )
 
        self.results = minimize(
            fun=portfolio_reverse_returns,
            x0=weights,
            method='SLSQP',
            bounds=selections,
            constraints=constraints
            )
        
    def get_optimization_results(self):
        min_w = self.results.x
        self.series = pd.Series(min_w, index=self.avg_daily_returns.index)
        self.optim_sharpe = np.dot(min_w, self.avg_daily_returns)/np.sqrt(np.dot(min_w, np.dot(min_w, self.covariance))*252)*100
        self.optim_returns = np.dot(min_w, self.avg_daily_returns)*100
        self.optim_vola = np.sqrt(np.dot(min_w, np.dot(self.covariance,min_w))*252)

def mc_simulator(portfolio: Portfolio, days: int):
    number_sims = 200
    portfolio_paths = pd.DataFrame(index=range(1,days+1))
    for n in range(1, number_sims+1):
        mu, sigma = portfolio.optim_returns/100, portfolio.optim_vola 
        returns = [1]
        for i in range(1,days):
            draw = np.random.normal(mu, sigma, 1)[0]
            returns.append(returns[i-1]*(1 + draw))
        portfolio_paths[f"Sim_{n}"] = returns

    return portfolio_paths



    
    

    