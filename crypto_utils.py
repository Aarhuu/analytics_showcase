import re

import pandas as pd
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

from keras.models import Sequential
#from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout



class DataService():
    def __init__(self, api_url):
        self.api_url: str = api_url
        self.scaler = None

    def get_chain_data(self):
        chains = requests.get(api_url + "chains")
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
    
    def create_master_candle_df(self, ohlcv_dict):
        columns = list(ohlcv_dict[list(ohlcv_dict.keys())[0]].columns)
        master_df = pd.DataFrame(data={}, index=ohlcv_dict[list(ohlcv_dict.keys())[0]].index.values)
        for key in ohlcv_dict.keys():
            for col in columns:
                master_df[str(key) + "_" + str(col)] = ohlcv_dict[key][col]
        
        if master_df.isnull().any().any():
            print("Candle data includes NaN values! Filling NaNs with either ffill or bfill")
            master_df = master_df.fillna(method="ffill").fillna("bfill")
        return master_df
    
    def scale_data(self, data, scaler=MinMaxScaler(feature_range=(0,1))):
        self.scaler = scaler
        return scaler.fit_transform(data)
    
    def get_X_Y(self, dataset, target_columns):
        #X  = dataset.drop(columns = target_columns)
        #Y = dataset[target_columns]
        X = np.delete(dataset, range(-len(target_columns), 0), 1)
        Y = dataset[:, range(-len(target_columns), 0)]
        return X, Y

    def add_lookback_columns(self, dataset, target_columns, look_back=1):
        Y = dataset[target_columns].shift(look_back)
        colnames = {col: col + f"_shifted{look_back}" for col in target_columns}
        Y = Y.rename(columns=colnames)
        return pd.concat([dataset, Y], axis=1).fillna(method="backfill")
        
    def create_simple_rnn(self, dense_units, hidden_units, input_shape, activations):
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

    def rescale(self, data):
        return self.scaler.inverse_transform(data)

    def get_plot_true_preds(self, preds, trues):
        fig = plt.figure(figsize=(20,10))
        plt.plot(trues, linestyle="dotted")
        plt.plot(preds)
        return fig

if __name__ == "__main__":
    api_url = "https://tradingstrategy.ai/api/"
    service = DataService(api_url=api_url)
    chain_data = service.get_chain_data()
    selected_chain = chain_data.iloc[0]["slug"]
    exchange_data = service.get_exchange_data(selected_chain=selected_chain)
    pairs_data = service.get_pairs_data(
        exc_slugs=exchange_data.iloc[0:3]["name"],
        chain_slugs=[selected_chain],
        n_pairs=3)
    start_time = "2024-01-01"
    end_time = "2024-01-15"
    candle_dict = service.get_ohlcv_candles(list(pairs_data.index.values), start_time, end_time, time_bucket="15m")
    master_df = service.create_master_candle_df(candle_dict)
    target_columns = ["1_Close"]
    master_df_shifted = service.add_lookback_columns(master_df, target_columns=target_columns, look_back=5)
    scaled = service.scale_data(master_df_shifted)
    X, y = service.get_X_Y(scaled, target_columns=target_columns)
    X_scaled, y_scaled = service.scale_data(X), service.scale_data(y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_train = np.reshape(y_train, (y_train.shape[0],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))

    simple_rnn = service.create_simple_rnn(len(target_columns), 50, (X_train.shape[1],1), ["relu", "linear"])
    simple_rnn.fit(X_train, y_train, epochs=10, batch_size=10)
    simple_rnn.summary()
    y_preds_train = simple_rnn.predict(X_train)
    y_preds_test = simple_rnn.predict(X_test)

    y_train_rescaled, y_test_rescaled = service.rescale(y_train), service.rescale(y_test)
    y_train_preds_rescaled, y_test_rescaled = service.rescale(y_preds_train), service.rescale(y_preds_test)

    
    y_trues_total = np.concatenate((y_train_rescaled, y_test_rescaled), axis=0)
    y_preds_total = np.concatenate((y_train_preds_rescaled, y_test_rescaled), axis=0)
    fig = service.get_plot_true_preds(y_preds_total, y_trues_total)
    fig.savefig("plots/simple_rnn.png")
