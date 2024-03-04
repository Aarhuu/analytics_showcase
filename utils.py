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
                 grid_cost,
                 daily_avg_energy_use, 
                 storage_capacity,
                 max_storage_duration = 2,
                 full_cap_discharge = 0.1) -> None:
        self.panel_area: float = panel_area
        self.grid_cost: float = grid_cost
        self.avg_energy_use: float = daily_avg_energy_use
        self.storage_capacity: float = storage_capacity
        self.max_storage_duration: float = max_storage_duration
        self.full_cap_discharge: float = full_cap_discharge
        self.spot_data: pd.DataFrame = pd.DataFrame()
        self.optim_h_bands: list = [0.0]
        self.optim_l_bands: list = [0.0]
        self.panel_efficiency: float = 0.8
        self.solar_to_electricity: float = 0.27777
        self.electricity_tax: float = 2.253
        self.simulated_savings: list = [0.0]
        self.simulated_storage: list = [0.0]
    
    def get_spot_data(self, filepath):
        self.spot_data = pd.read_csv(filepath)
        self.spot_data["Aika"] = pd.to_datetime(self.spot_data["Aika"], format="%Y-%m-%d %H:%M")
    
    def simulate_system(self, step_size):
        sunlight_hours = range(10, 17)

        def get_action(initial_storage, hour):

            if (self.spot_data.iloc[-1, 0] - self.spot_data.iloc[hour, 0]).days < 1:
                day_ahead_data = self.spot_data.iloc[hour:,:]
            else:
                day_ahead_data = self.spot_data.iloc[hour:hour+24,:]  

            l_band = day_ahead_data["c/kWh"].min()
            h_band = day_ahead_data["c/kWh"].max()      
            best_result = 0.0

            optim_h_band = h_band
            optim_l_band = l_band

            while h_band >= day_ahead_data["c/kWh"].min():
                while l_band <= h_band:
                    savings = [0.0]
                    storage = [initial_storage]
                    for h in range(0, day_ahead_data.shape[0]):
                        month = day_ahead_data.iloc[h, 0].month
                        hour_of_day = day_ahead_data.iloc[h, 0].hour
                        price = day_ahead_data.iloc[h, 1]
                        if price < l_band and storage[h-1] < self.storage_capacity and hour_of_day in sunlight_hours:
                            available_solar = daily_avg_solar[str(month)]/len(sunlight_hours)*self.panel_area*self.panel_efficiency*self.solar_to_electricity
                            if available_solar < self.storage_capacity - storage[h-1]:
                                storage.append(storage[h-1] + available_solar)
                                savings.append(savings[h-1] + (self.avg_energy_use/24)*(price + self.grid_cost + self.electricity_tax)/100)

                            else:
                                storage.append(self.storage_capacity)
                                savings.append(savings[h-1] + (self.avg_energy_use/24 - (storage[h] - storage[h-1]))*(price + self.grid_cost + self.electricity_tax)/100)

                        elif price > h_band and storage[h-1] > 0.0:
                            used_storage_energy = np.min([self.avg_energy_use/24.0, storage[h-1]])
                            storage.append(storage[h-1] - used_storage_energy)
                            savings.append(savings[h-1] - used_storage_energy*(price + self.grid_cost + self.electricity_tax)/100)

                        elif storage[h-1] == self.storage_capacity:
                            used_storage_energy = np.min([self.avg_energy_use/24.0, storage[h-1]*self.full_cap_discharge])
                            storage.append(storage[h-1] - used_storage_energy)
                            savings.append(savings[h-1] - used_storage_energy*(price + self.grid_cost + self.electricity_tax)/100)
                        else:
                            storage.append(storage[h-1]*0.99)
                            savings.append(savings[h-1])
                    
                    if -1*savings[-1] > best_result: 
                        best_result = -1*savings[-1]
                        optim_h_band = h_band
                        optim_l_band = l_band
                    l_band += step_size
                l_band = day_ahead_data["c/kWh"].min()
                h_band -= step_size
            return optim_h_band, optim_l_band

      

        for hour in range(1, len(self.spot_data)):
            
            optim_h_band, optim_l_band = get_action(self.simulated_storage[hour-1], hour)
            self.optim_h_bands.append(optim_h_band)
            self.optim_l_bands.append(optim_l_band)
            hour_of_day = self.spot_data.iloc[hour, 0].hour
            month = self.spot_data.iloc[hour, 0].month
            price = self.spot_data.iloc[hour, 1]
            if self.spot_data.iloc[hour, 1] < optim_l_band and self.simulated_storage[hour-1] < self.storage_capacity and hour_of_day in sunlight_hours:
                available_solar = daily_avg_solar[str(month)]/len(sunlight_hours)*self.panel_area*self.panel_efficiency*self.solar_to_electricity
                if available_solar < self.storage_capacity - self.simulated_storage[hour-1]:
                    self.simulated_storage.append(self.simulated_storage[hour-1] + available_solar)
                    self.simulated_savings.append(self.simulated_savings[hour-1] + (self.avg_energy_use/24)*(price + self.grid_cost + self.electricity_tax)/100)

                else:
                    self.simulated_storage.append(self.storage_capacity)
                    self.simulated_savings.append(self.simulated_savings[hour-1] + (self.avg_energy_use/24 - (self.simulated_storage[hour] - self.simulated_storage[hour-1]))*(price + self.grid_cost + self.electricity_tax)/100)

            elif price > optim_h_band and self.simulated_storage[hour-1] > 0.0:
                used_storage_energy = np.min([self.avg_energy_use/24.0, self.simulated_storage[hour-1]])
                self.simulated_storage.append(self.simulated_storage[hour-1] - used_storage_energy)
                self.simulated_savings.append(self.simulated_savings[hour-1] - used_storage_energy*(price + self.grid_cost + self.electricity_tax)/100)

            elif self.simulated_storage[hour-1] == self.storage_capacity:
                used_storage_energy = np.min([self.avg_energy_use/24.0, self.simulated_storage[hour-1]*self.full_cap_discharge])
                self.simulated_storage.append(self.simulated_storage[hour-1] - used_storage_energy)
                self.simulated_savings.append(self.simulated_savings[hour-1] - used_storage_energy*(price + self.grid_cost + self.electricity_tax)/100)
            else:
                self.simulated_storage.append(self.simulated_storage[hour-1]*0.99)
                self.simulated_savings.append(self.simulated_savings[hour-1])

if __name__ == "__main__":
    energy_system = EnergySystem(
        panel_area=50, 
        grid_cost= 5.0,
        daily_avg_energy_use=30, 
        storage_capacity=20)
    energy_system.get_spot_data("./Oomi Spot-hintatieto.csv")
    energy_system.simulate_system(step_size=3.5)
    print(energy_system.simulated_storage)

    
    

    