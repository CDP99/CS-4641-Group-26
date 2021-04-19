import os
import requests
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

import statsmodels.tsa.api as tsa
from scipy.stats import probplot, moment

import matplotlib.pyplot as plt
import seaborn as sns

from api_keys import fmp_api_key as api_key
from API import API

api = API(api_key, period='quarter', limit=None)
tickers = api.getRandTickers(50)

X = api.getSamplesFromTickers(tickers, maxTickers=5)
Y = api.getYFromDF(X)

merge = pd.merge(X, Y, how='inner', left_on=["date", "symbol"], right_on=["date", "symbol"])
X = merge.drop(['date', 'symbol', 'futureDate', 'futureClose'], axis=1).to_numpy()
Y = merge[['futureClose']].to_numpy()

print()