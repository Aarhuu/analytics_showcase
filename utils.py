import datetime
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
        self.covariance = self.daily_returns.cov()
        volas = {}
        for _, x in enumerate(self.close_data.columns.values):
            volas[x] = np.sqrt(self.trading_days)*self.daily_returns[x].std()
        self.volatility = pd.Series(data=volas)

    def optimize(self) -> None:

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
        self.optim_returns = self.avg_daily_returns.dot(min_w)
        print(min_w)
        print(self.avg_daily_returns)
        self.optim_vola = np.sqrt(np.dot(min_w, np.dot(min_w, self.covariance)))
        print(self.volatility)
        print(self.covariance)
        print(self.optim_vola)

    
"""
class EnergySystem

    - panel area of local solar power plant in square meters
    - daily_avg_energy use in kWh
    - storage capacity in kWh
    - max storage_duration in hours (e.g. 48 hours equals two days)
    - spot_data from Oomi energia website

"""

# per day average per month MJ/m2 in Finland
daily_avg_solar = {
	"1": 32/31,
	"2": 90/28,
	"3": 244/31,
	"4": 403/30,	
	"5": 592/31,	
	"6": 615/30,	
	"7": 615/31,	
	"8": 470/31,	
	"9": 270/30,	
	"10": 120/31,	
	"11": 35/30,	
	"12": 16/31,	
}

class EnergySystem():
    def __init__(self, 
                 panel_area, 
                 daily_avg_energy_use, 
                 storage_capacity,
                 max_storage_duration,
                 full_cap_discharge) -> None:
        self.panel_area: float = panel_area
        self.avg_energ_use: float = daily_avg_energy_use
        self.storage_capacity: float = storage_capacity
        self.max_storage_duration: float = max_storage_duration
        self.full_cap_discharge: float = full_cap_discharge
        self.spot_data: pd.DataFrame = pd.DataFrame()
        self.h_band: float = None
        self.l_band: float = None
        self.panel_efficiency: float = 0.22
        self.solar_to_electricity: float = 0.27777
        self.simulated_savings: list = None
        self.simulated_storage: list = None
    
    def get_spot_data(self, filepath):
        self.spot_data = pd.read_csv(filepath)
        self.spot_data["Aika"] = pd.to_datetime(self.spot_data["Aika"], format="%Y-%m-%d %H:%M")
    def set_bands(self, h_band, l_band):
        self.h_band, self.l_band = h_band, l_band

    def simulate_system(self):
        savings = [0.0]
        storage = [0.0]
        for h in range(1, self.spot_data.shape[0]):
            month = self.spot_data.iloc[h, 0].month
            hour_of_day = self.spot_data.iloc[h, 0].hour
            price = self.spot_data.iloc[h, 1]
            if price < self.l_band and storage[h-1] < self.storage_capacity:
                available_solar = daily_avg_solar[str(month)]*self.panel_area*self.panel_efficiency*self.solar_to_electricity
                if available_solar*self.panel_efficiency < self.storage_capacity - storage[h-1]:
                    stored = available_solar*self.panel_efficiency
                    storage.append(storage[h-1] + stored)
                    savings.append(savings[h-1] + (available_solar*price/100))
                else:
                    storage.append(self.storage_capacity)
                    savings.append(savings[h-1] + (storage[h] - storage[h-1])*price/100)

            elif price > self.h_band and storage[h-1] != 0.0 and str(month):
                available_energy = storage[h-1]
                used_storage_energy = np.min([self.avg_energ_use, available_energy])
                storage.append(storage[h-1] - used_storage_energy)
                savings.append(savings[h-1] - price*used_storage_energy/100)
            elif storage[h-1] == self.storage_capacity:
                storage.append(storage[h-1]*(1 - self.full_cap_discharge))
                savings.append(savings[h-1] - price*storage[h-1]*(1 - self.full_cap_discharge)/100)    

            else:
                storage.append(storage[h-1])
                savings.append(savings[h-1])
                        
        self.simulated_storage = storage
        self.simulated_savings = savings



if __name__ == "__main__":
    energy_system = EnergySystem(
        panel_area=15, 
        daily_avg_energy_use=30, 
        storage_capacity=60, 
        max_storage_duration=2,
        full_cap_discharge=0.1)
    energy_system.get_spot_data("./Oomi Spot-hintatieto.csv")
    energy_system.set_bands(energy_system.spot_data["c/kWh"].mean()*1.5, energy_system.spot_data["c/kWh"].mean()*0.5)
    energy_system.simulate_system()
    
        




    
    

    