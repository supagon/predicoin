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
    btcinfo = pd.read_csv('btc.csv')
    btcdaily = pd.read_csv('btc.csv',header=0)
    btcdaily['Date'] = pd.to_datetime(btcdaily['Date'])
    btcdaily = btcdaily.resample('D', on='Date').sum()
    btcdaily['24hchange'] = btcdaily['Adj Close'].pct_change()
    btcinfo = pd.merge(btcinfo,btcdaily[['24hchange']],how='outer' ,on=btcinfo['Date'])
    btcinfo = btcinfo.drop(columns={'key_0'})
    print(btcinfo)
    json_result = btcinfo.astype(str).to_json(orient='index')
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
    json_result =  btcsma.astype(str).to_json(orient='index')
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

#MACD
btcmacd = pd.read_csv('btc.csv').set_index('Date')
btcmacd.index =pd.to_datetime(btcmacd.index)

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'MACD'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'MACD':'Signal line'})
    hist = pd.DataFrame(macd['MACD'] - signal['Signal line']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

btc_macd = get_macd(btcmacd['Close'], 26, 12, 9)

def implement_macd_strategy(prices, data):    
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(data)):
        if data['MACD'][i] > data['Signal line'][i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        elif data['MACD'][i] < data['Signal line'][i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
            
    return buy_price, sell_price, macd_signal
            
buy_price, sell_price, macd_signal = implement_macd_strategy(btcmacd['Close'],btc_macd)

position = []
for i in range(len(macd_signal)):
    if macd_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(btcmacd['Close'])):
    if  macd_signal[i] == 1:
        position[i] = 1
    elif macd_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
macd = btc_macd['MACD']
signal = btc_macd['Signal line']
close_price = btcmacd['Close']
macd_signal = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(btcmacd.index)
position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(btcmacd.index)

macdframes = [close_price, macd, signal, macd_signal, position]
macdstrategy = pd.concat(macdframes, join = 'inner', axis = 1)
#print(macdstrategy)

btcmacd = pd.merge(btcmacd,macdstrategy[['macd_signal','macd_position']],how='outer' ,on=btcmacd.index)
btcmacd = btcmacd.rename(columns={'key_0':'Date'})
btcmacd = pd.merge(btcmacd,btc_macd[['MACD','Signal line','hist']],how='outer' ,on='Date')

def get_btc_macd():
    json_macd =  btcmacd.astype(str).to_json(orient='index')
    btcmacdName = "["+json_macd+","+"{\"name\": \"BTC\"}]"
    return btcmacdName

def post_btc_macd(investment_value):
    btcmacd_ret = pd.DataFrame(np.diff(btcmacd['Close'])).rename(columns = {0:'returns'})
    macd_strategy_ret = []

    for i in range(len(btcmacd_ret)):
        try:
            returns = btcmacd_ret['returns'][i]*macdstrategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass
        
    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns = {0:'macd_returns'})

    #investment_value = 100000
    number_of_stocks = floor(investment_value/btcmacd['Close'][0])
    macd_investment_ret = []

    for i in range(len(macd_strategy_ret_df['macd_returns'])):
        returns = number_of_stocks*macd_strategy_ret_df['macd_returns'][i]
        macd_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    #profit_percentage = floor((total_investment_ret/investment_value)*100)
    return(f'{total_investment_ret}')