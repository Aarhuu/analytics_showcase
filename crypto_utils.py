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
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

api_url = "https://tradingstrategy.ai/api/"

def get_chain_data(api_url):

    chains = requests.get(api_url + "chains")
    chain_names = [chain["chain_name"] for chain in chains.json()]
    chain_slugs = [chain["chain_slug"] for chain in chains.json()]
    chain_slug_string = ",".join(chain_slugs)
    return chain_slugs
    selected_chain = chain_slugs[0]