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

eth = web.get_data_yahoo('ETH-USD', start='2018-01-01', end=datetime.now())

eth.to_csv("eth.csv")
eth = pd.read_csv("eth.csv")


def get_eth_info():
    ethinfo = pd.read_csv('eth.csv')
    ethdaily = pd.read_csv('eth.csv',header=0)
    ethdaily['Date'] = pd.to_datetime(ethdaily['Date'])
    ethdaily = ethdaily.resample('D', on='Date').sum()
    ethdaily['24hchange'] = ethdaily['Adj Close'].pct_change()
    ethinfo = pd.merge(ethinfo,ethdaily[['24hchange']],how='outer' ,on=ethinfo['Date'])
    ethinfo = ethinfo.drop(columns={'key_0'})
    #print(ethinfo)
    json_result = ethinfo.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"ETH\"}]"
    return addName