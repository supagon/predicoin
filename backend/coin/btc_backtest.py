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

btc = web.get_data_yahoo('BTC-USD', start='2018-01-01', end=datetime.now())

btc.to_csv("btc.csv")
btc = pd.read_csv("btc.csv")

def get_btc_info():
    json_result = btc.to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"BTC\"}]"
    return addName

btcsma = pd.read_csv('btc.csv').set_index('Date')
btcsma.index =pd.to_datetime(btcsma.index)
btcsma.tail

def btc_sma(data, n):
    sma = data.rolling(window = n).mean()
    return pd.DataFrame(sma)

n = [20, 50]
for i in n:
    btcsma[f'sma_{i}'] = btc_sma(btcsma['Close'], i)
    
def implement_sma_strategy(data, short_window, long_window):
    sma1 = short_window
    sma2 = long_window
    buy_price = []
    sell_price = []
    sma_signal = []
    signal = 0
    
    for i in range(len(data)):
        if sma1[i] > sma2[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                sma_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                sma_signal.append(0)
        elif sma2[i] > sma1[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                sma_signal.append(-1)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                sma_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            sma_signal.append(0)
            
    return buy_price, sell_price, sma_signal

sma_20 = btcsma['sma_20']
sma_50 = btcsma['sma_50']

buy_price, sell_price, signal = implement_sma_strategy(btcsma['Close'], sma_20, sma_50)

position = []
for i in range(len(signal)):
    if signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(btcsma['Close'])):
    if signal[i] == 1:
        position[i] = 1
    elif signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]

sma_20 = pd.DataFrame(sma_20).rename(columns = {0:'sma_20'})
sma_50 = pd.DataFrame(sma_50).rename(columns = {0:'sma_50'}) 
signal = pd.DataFrame(signal).rename(columns = {0:'sma_signal'}).set_index(btcsma.index)
position = pd.DataFrame(position).rename(columns = {0:'sma_position'}).set_index(btcsma.index)

frames = [sma_20, sma_50, signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)
strategy = strategy.reset_index().drop('Date', axis = 1)



btcsma = pd.merge(btcsma,strategy[['sma_signal','sma_position']],how='outer' ,on=btcsma.index)
btcsma = btcsma.rename(columns={'key_0':'Date'})

def get_btc_sma():
    json_result =  btcsma.to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"BTC\"}]"
    return addName

def post_btc_sma(investment_value):
    btc_ret = pd.DataFrame(np.diff(btcsma['Close'])).rename(columns = {0:'returns'})
    sma_strategy_ret = []

    for i in range(len(btc_ret)):
        try:
            returns = btc_ret['returns'][i]*strategy['sma_position'][i]
            sma_strategy_ret.append(returns)
        except:
            pass
        
    sma_strategy_ret_df = pd.DataFrame(sma_strategy_ret).rename(columns = {0:'sma_returns'})

    number_of_stocks = math.floor(investment_value/btcsma['Close'][1])
    sma_investment_ret = []

    for i in range(len(sma_strategy_ret_df['sma_returns'])):
        returns = number_of_stocks*sma_strategy_ret_df['sma_returns'][i]
        sma_investment_ret.append(returns)

    sma_investment_ret_df = pd.DataFrame(sma_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(sma_investment_ret_df['investment_returns']), 2)
    return(f'{total_investment_ret}')