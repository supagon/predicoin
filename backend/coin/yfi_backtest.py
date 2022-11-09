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

yfi = web.get_data_yahoo('YFI-USD', start='2021-01-01', end=datetime.now())

yfi.to_csv("yfi.csv")
yfi = pd.read_csv("yfi.csv")


def get_yfi_info():
    yfiinfo = pd.read_csv('yfi.csv')
    yfidaily = pd.read_csv('yfi.csv',header=0)
    yfidaily['Date'] = pd.to_datetime(yfidaily['Date'])
    yfidaily = yfidaily.resample('D', on='Date').sum()
    yfidaily['24hchange'] = yfidaily['Adj Close'].pct_change()
    yfiinfo = pd.merge(yfiinfo,yfidaily[['24hchange']],how='outer' ,on=yfiinfo['Date'])
    yfiinfo = yfiinfo.drop(columns={'key_0'})
    #print(yfiinfo)
    json_result = yfiinfo.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"YFI\"}]"
    return addName