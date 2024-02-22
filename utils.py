from datetime import date

import pandas as pd
import yfinance as yf
import numpy as np
import scipy
from scipy.optimize import minimize

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

    def download_close_data(self) -> None:
        self.close_data = yf.download(self.tickers, self.start_date, self.end_date).Close
        self.trading_days = self.close_data.shape[0]

    def get_daily_returns(self):
        for _, x in enumerate(self.close_data.columns.values):
            self.daily_returns[x] = self.close_data[x].pct_change().replace("", None).ffill()

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
        volas = {}
        for _, x in enumerate(self.close_data.columns.values):
            volas[x] = np.sqrt(self.trading_days)*self.daily_returns[x].std() 
        self.volatility = pd.Series(data=volas)

    def optimize(self) -> None:
        self.covariance = self.daily_returns.cov()
        def portfolio_returns(w):
            #annualized_returns = np.dot(np.transpose(weights), self.avg_daily_returns)*252
            annualized_returns = np.sqrt(np.dot(w, np.dot(w, self.covariance)))/(self.avg_daily_returns*252).dot(w)
            return annualized_returns
        
        weights = [1/len(self.tickers)]*len(self.tickers)
        selections = tuple((0, 1) for i in range(len(self.tickers)))
        constraints = (
        {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
        )
 
        self.results = minimize(
            fun=portfolio_returns,
            x0=weights,
            bounds=selections,
            constraints=constraints
            )
        
    def get_optimization_results(self):
        min_w = self.results.x
        self.optim_returns = self.avg_daily_returns.dot(min_w)*100*252
        print(self.avg_daily_returns)
        print(self.optim_returns)
        self.optim_vola = np.sqrt(np.dot(min_w, np.dot(min_w, self.covariance)))

    
    