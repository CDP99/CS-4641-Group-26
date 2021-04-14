import os
import requests
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

import statsmodels.tsa.api as tsa
from scipy.stats import probplot, moment

import matplotlib.pyplot as plt
import seaborn as sns

from api_keys import fmp_api_key as api_key
from API import API

api = API(api_key, period='quarter', limit=52)
res = api.getRatios("AAPL")

# display(res)
datetime_series = pd.to_datetime(res['date'])
datetime_index = pd.DatetimeIndex(datetime_series.values)
df=res.set_index(datetime_index)
df.drop('date',axis=1,inplace=True)
features = [
    "currentRatio", "quickRatio", "grossProfitMargin", "operatingProfitMargin", "returnOnAssets",
    "returnOnEquity", "returnOnCapitalEmployed", "debtEquityRatio", "priceToBookRatio", "priceToSalesRatio",
    "priceEarningsRatio", "priceEarningsToGrowthRatio", "priceToOperatingCashFlowsRatio" 
]
df = df.squeeze().dropna()

ratios = [
    "currentRatio", "quickRatio", "debtEquityRatio", "priceToBookRatio", "priceToSalesRatio",
    "priceEarningsRatio", "priceEarningsToGrowthRatio", "priceToOperatingCashFlowsRatio" 
]
percentages = [
    "grossProfitMargin", "operatingProfitMargin", "returnOnAssets",
    "returnOnEquity", "returnOnCapitalEmployed"
]

# min_max_scaler = MinMaxScaler((-1, 1))
# df[percentages] = min_max_scaler.fit_transform(df[percentages])  # No need, because percentages are already on a good scale
ss = StandardScaler()
# standard_scaler.fit(df.loc[:, df.columns != 'symbol'])
df_without_symbol = df.loc[:, df.columns != 'symbol']
df_scaled = pd.DataFrame(ss.fit_transform(df_without_symbol), columns = df_without_symbol.columns).set_index(res['date'])
df_scaled = pd.concat(df['symbol'], df_without_symbol)