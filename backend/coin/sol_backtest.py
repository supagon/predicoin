import pandas_datareader.data as web
import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pandas_datareader.data import DataReader
import sklearn.preprocessing
from scipy.signal import savgol_filter
import math
from math import floor
from termcolor import colored as cl
plt.style.use('fivethirtyeight')
# %matplotlib inline

sol = web.get_data_yahoo('SOL-USD', start='2021-01-01', end=datetime.now())

sol.to_csv("sol.csv")
sol = pd.read_csv("sol.csv")


def get_sol_info():
    solinfo = pd.read_csv('sol.csv')
    soldaily = pd.read_csv('sol.csv',header=0)
    soldaily['Date'] = pd.to_datetime(soldaily['Date'])
    soldaily = soldaily.resample('D', on='Date').sum()
    soldaily['24hchange'] = soldaily['Adj Close'].pct_change()
    solinfo = pd.merge(solinfo,soldaily[['24hchange']],how='outer' ,on=solinfo['Date'])
    solinfo = solinfo.drop(columns={'key_0'})
    #print(solinfo)
    json_result = solinfo.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"SOL\"}]"
    return addName