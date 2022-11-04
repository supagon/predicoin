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

paxg = web.get_data_yahoo('PAXG-USD', start='2018-01-01', end=datetime.now())

paxg.to_csv("paxg.csv")
paxg = pd.read_csv("paxg.csv")


def get_paxg_info():
    paxginfo = pd.read_csv('paxg.csv')
    paxgdaily = pd.read_csv('paxg.csv',header=0)
    paxgdaily['Date'] = pd.to_datetime(paxgdaily['Date'])
    paxgdaily = paxgdaily.resample('D', on='Date').sum()
    paxgdaily['24hchange'] = paxgdaily['Adj Close'].pct_change()
    paxginfo = pd.merge(paxginfo,paxgdaily[['24hchange']],how='outer' ,on=paxginfo['Date'])
    paxginfo = paxginfo.drop(columns={'key_0'})
    print(paxginfo)
    json_result = paxginfo.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"PAXG\"}]"
    return addName