import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# from keras.models import Sequential
# #from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import SimpleRNN
# from keras.layers import Dropout

class Visualizer():
    def __init__(self) -> None:
        pass

    def get_line_plot(self, data, title = "example plot"):
        columns = data.columns
        fig, ax = plt.subplots()
        for col in columns:
            ax.plot(data[col], label=col)
        ax.set_title(f"Series data from {data.index.values[0]} - {data.index.values[-1]}")
        plt.legend(loc="best")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(title)
        return ax
    
    def get_box_plot(self, data):
        fig, ax = plt.subplots()
        sns.boxplot(data=data, ax=ax)
        ax.set_title(f"Box plot")
        plt.legend(loc="best")
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel("Pair price")
        return ax

    def get_vol_bar_counts(self, vol_bar_df, freq="15min"):
        return vol_bar_df.groupby(pd.Grouper(freq=freq)).first().iloc[:,0:].fillna(0).plot()

class DataService():
    def __init__(self, api_url):
        self.api_url: str = api_url
        self.scaler = None

    def get_chain_data(self):
        chains = requests.get(self.api_url + "chains")
        chain_names = [chain["chain_name"] for chain in chains.json()]
        chain_slugs = [chain["chain_slug"] for chain in chains.json()]
        chain_id = [chain["chain_id"] for chain in chains.json()]
        chain_data = pd.DataFrame(data={
            "name" : chain_names,
            "slug": chain_slugs},
            index=chain_id)
        return chain_data

    def get_exchange_data(self, selected_chain: str, filter_zero_volume = "true"):
        api_extension = f"exchanges?chain_slug={selected_chain}&sort=usd_volume_30d&direction=desc&filter_zero_volume={filter_zero_volume}"
        exchanges = requests.get(self.api_url + api_extension)
        exc_names = [exc["exchange_slug"]for exc in exchanges.json()["exchanges"]]
        exc_vol = [exc["usd_volume_30d"]for exc in exchanges.json()["exchanges"]]
        exc_id = [exc["exchange_id"]for exc in exchanges.json()["exchanges"]]
        exchange_data = pd.DataFrame(data={
            "name" : exc_names,
            "volume": exc_vol},
            index=exc_id)
        return exchange_data
    
    def get_pairs_data(self, exc_slugs: list[str], chain_slugs: list[str], n_pairs: int, sort="volume_30d", filter="min_liquidity_1M"):
        exc_slug_string = ",".join(exc_slugs)
        chain_slug_string = ",".join(chain_slugs)
        api_extension = f"pairs?exchange_slugs={exc_slug_string}&chain_slugs={chain_slug_string}&page=0&page_size={n_pairs}&sort={sort}&direction=desc&filter={filter}&eligible_only=true&format=json"
        pairs = requests.get(self.api_url + api_extension)
        pair_id = [pair["pair_id"] for pair in pairs.json()["results"]]
        pair_slug = [pair["pair_slug"] for pair in pairs.json()["results"]]
        pair_exc = [pair["exchange_slug"] for pair in pairs.json()["results"]]
        pair_data = pd.DataFrame(data={
            "pair_slug" : pair_slug,
            "pair_exchange": pair_exc}, 
            index=pair_id)
        return pair_data
    
    def get_ohlcv_candles(self, pair_ids: list[int|str], start_time, end_time,  time_bucket="15m"):
        pair_ids_string = ",".join(map(str, pair_ids))
        api_extension = f"candles?pair_ids={pair_ids_string}&time_bucket={time_bucket}&candle_type=price&start={start_time}&end={end_time}"
        response = requests.get(self.api_url + api_extension)
        values = response.json()
        candle_dict = {}
        for key, value in values.items():
            ts = [val["ts"] for val in value]
            open = [val["o"] for val in value]
            high = [val["h"] for val in value]
            low = [val["l"] for val in value]
            close = [val["c"] for val in value]
            volume = [val["v"] for val in value]
            df = pd.DataFrame(data={"Open": open, "High": high, "Low": low, "Close": close, "Volume": volume}, index=pd.to_datetime(ts, format='%Y-%m-%dT%H:%M:%S'))
            candle_dict[key] = df
        return candle_dict
    
    
    def get_volume_bars(self, data, m):

        def get_volume_indxs(col):
            t = data[col]
            ts = 0
            idx = []
            for i, x in enumerate(t):
                ts += x
                if ts >= m:
                    idx.append(i)
                    ts = 0
                    continue
            return idx
        
        volume_df = pd.DataFrame()
        volume_columns = [col for col in data.columns if "Volume" in col]
        for column in volume_columns:
            indx = get_volume_indxs(column)
            colname = column.replace("Volume", "Close")
            volume_df[colname] = data.iloc[indx][column].drop_duplicates()
        
        return volume_df
    
    

    def create_master_candle_df(self, ohlcv_dict):
        columns = list(ohlcv_dict[list(ohlcv_dict.keys())[0]].columns)
        master_df = pd.DataFrame(data={}, index=ohlcv_dict[list(ohlcv_dict.keys())[0]].index.values)
        for key in ohlcv_dict.keys():
            for col in columns:
                master_df[str(key) + "_" + str(col)] = ohlcv_dict[key][col]
        
        if master_df.isnull().any().any():
            print("Candle data includes NaN values! Filling NaNs with either ffill or bfill")
            master_df = master_df.ffill().bfill()
        return master_df
    
    def get_close_prices(self, dataset):
        close_columns = [col for col in dataset.columns if "Close" in col]
        return dataset[close_columns]
    
    def get_log_returns(self, data):
        log_returns = pd.DataFrame()
        for col in data.columns:
            log_returns[col] = np.log(data[col]/data[col].shift(1))
        return log_returns
    
    def scale_data(self, data, scaler=MinMaxScaler(feature_range=(0,1))):
        self.scaler = scaler
        scaled = self.scaler.fit_transform(data)
        scaled = pd.DataFrame(data=scaled,
                              index=data.index,
                              columns = data.columns)
        return scaled
    
    def remove_outliers(self, data, threshold=3):
        return data[(np.abs(stats.zscore(data)) < threshold).all(axis=1)]
    
    def get_serial_correlation(self, data):
        corrs = pd.DataFrame(data={"autocorr": [0]*len(data.columns), "number_of_samples": [0]*len(data.columns)} ,index=data.columns)
        
        for col in data.columns:
           corrs.loc[col, "autocorr"] = pd.Series.autocorr(data[col])
           corrs.loc[col, "number_of_samples"] = data[col].shape[0]
        
        return corrs
    
    def get_X_Y(self, dataset, target_columns):
        data_array = dataset.to_numpy()
        X = np.delete(data_array, range(-len(target_columns), 0), 1)
        Y = data_array[:, range(-len(target_columns), 0)]
        return X, Y

    def add_lookback_columns(self, dataset, look_back=1):
        shifted = dataset.shift(look_back)
        colnames = {col: col + f"_shifted_{look_back}" for col in shifted.columns}
        shifted = shifted.rename(columns=colnames).bfill()
        return shifted

    def add_signal_column(self, dataset):
        return [1 if x > 0 else 0 for x in dataset.diff().bfill()]
        
    def create_simple_rnn_regressor(self, dense_units, hidden_units, input_shape, activations):
        # initializing the RNN
        regressor = Sequential()
        
        # adding RNN layers and dropout regularization
        regressor.add(SimpleRNN(units = hidden_units, 
                                activation = activations[0],
                                return_sequences = True,
                                input_shape = input_shape))
        regressor.add(Dropout(0.2))
        
        regressor.add(SimpleRNN(units = hidden_units, 
                                activation = activations[0],
                                return_sequences = True))
        
        regressor.add(SimpleRNN(units = hidden_units, 
                                activation = activations[0],
                                return_sequences = True))
        
        regressor.add(SimpleRNN(units = hidden_units))
        
        # adding the output layer
        regressor.add(Dense(units = dense_units, activation=activations[1]))
        
        # compiling RNN
        regressor.compile(optimizer = "adam", 
                        loss = "mean_squared_error")
        
        return regressor
    

if __name__ == "__main__":
    api_url = "https://tradingstrategy.ai/api/"
    data_service = DataService(api_url=api_url)
    chain_data = data_service.get_chain_data()
    selected_chain = chain_data.iloc[0]["slug"]
    exchange_data = data_service.get_exchange_data(selected_chain=selected_chain)

    n_exchange = 3
    n_pairs = 3
    time_bucket = "1m"

    pairs_data = data_service.get_pairs_data(
        exc_slugs=exchange_data.iloc[0:n_exchange]["name"],
        chain_slugs=[selected_chain],
        n_pairs=n_pairs)
    start_time = "2024-01-01"
    end_time = "2024-01-02"
    candle_dict = data_service.get_ohlcv_candles(list(pairs_data.index.values), start_time, end_time, time_bucket=time_bucket)

    master_df = data_service.create_master_candle_df(candle_dict)
    close_prices = data_service.get_close_prices(master_df)
    vis = Visualizer()
    #vol_bar_df["1_Close"].groupby(pd.Grouper(freq = "5min")).count().plot()
    vol_bar_df = data_service.get_volume_bars(master_df, m=100).dropna()

    scaled_closes = data_service.scale_data(vol_bar_df)
    log_returns = data_service.get_log_returns(vol_bar_df)
    serial_autocorr = data_service.get_serial_correlation(log_returns)

    print(serial_autocorr)

    # vis = Visualizer()

    # log_returns = data_service.get_log_returns(close_prices)
    # scaled_closes = data_service.scale_data(close_prices)


#1_Close  3366033_Close     239_Close
