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

eth = web.get_data_yahoo('ETH-USD', start='2021-01-01', end=datetime.now())

eth.to_csv("eth.csv")
eth = pd.read_csv("eth.csv")

def max_drawdown(eth, window = 365):
    Roll_Max = eth['Close'].rolling(window, min_periods=1).max()
    Daily_Drawdown = eth['Close']/Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

    return Max_Daily_Drawdown

def get_eth_info():
    ethinfo = pd.read_csv('eth.csv')
    ethdaily = pd.read_csv('eth.csv',header=0)
    ethdaily['Date'] = pd.to_datetime(ethdaily['Date'])
    ethdaily = ethdaily.resample('D', on='Date').sum()
    ethdaily['24hchange'] = ethdaily['Adj Close'].pct_change()
    ethinfo = pd.merge(ethinfo,ethdaily[['24hchange']],how='outer' ,on=ethinfo['Date'])
    ethinfo = ethinfo.drop(columns={'key_0'})
    ethinfo['Max_dd'] = max_drawdown(pd.read_csv('eth.csv'))
    print(ethinfo)
    json_result = ethinfo.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"ETH\"}]"
    return addName

ethsma = pd.read_csv('eth.csv').set_index('Date')
ethsma.index =pd.to_datetime(ethsma.index)

def eth_sma(data, n):
    sma = data.rolling(window = n).mean()
    return pd.DataFrame(sma)

n = [20, 50]
for i in n:
    ethsma[f'sma_{i}'] = eth_sma(ethsma['Close'], i)
    
def eth_implement_sma_strategy(data, short_window, long_window):
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


sma_20 = ethsma['sma_20']
sma_50 = ethsma['sma_50']

buy_price, sell_price, signal = eth_implement_sma_strategy(ethsma['Close'], sma_20, sma_50)

position = []
for i in range(len(signal)):
    if signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(ethsma['Close'])):
    if signal[i] == 1:
        position[i] = 1
    elif signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]

sma_20 = pd.DataFrame(sma_20).rename(columns = {0:'sma_20'})
sma_50 = pd.DataFrame(sma_50).rename(columns = {0:'sma_50'}) 
signal = pd.DataFrame(signal).rename(columns = {0:'sma_signal'}).set_index(ethsma.index)
position = pd.DataFrame(position).rename(columns = {0:'sma_position'}).set_index(ethsma.index)

frames = [sma_20, sma_50, signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)
strategy = strategy.reset_index().drop('Date', axis = 1)



ethsma = pd.merge(ethsma,strategy[['sma_signal','sma_position']],how='outer' ,on=ethsma.index)
ethsma = ethsma.rename(columns={'key_0':'Date'})

def plt_eth_sma():
    ethsma.set_index('Date',inplace=True)
    sma_20 = ethsma['sma_20']
    sma_50 = ethsma['sma_50']
    buy_price, sell_price, signal = eth_implement_sma_strategy(ethsma['Close'], sma_20, sma_50)
    plt.plot(ethsma['Close'], alpha = 0.3, label = 'eth')
    plt.plot(sma_20, alpha = 0.6, label = 'SMA 20')
    plt.plot(sma_50, alpha = 0.6, label = 'SMA 50')
    plt.scatter(ethsma.index, buy_price, marker = '^', s = 200, color = 'darkblue', label = 'BUY SIGNAL')
    plt.scatter(ethsma.index, sell_price, marker = 'v', s = 200, color = 'crimson', label = 'SELL SIGNAL')
    plt.legend(loc = 'upper left')
    plt.title('eth SMA CROSSOVER TRADING SIGNALS')
    
    #date_form = DateFormatter("%d-%m-%Y")
    #plt.xaxis.set_major_formatter(date_form)
    #plt.show()
    #plt.savefig("output.jpg")
    return(plt.savefig("eth_sma_output.jpg"))

def get_eth_sma():
    json_result =  ethsma.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"ETH\"}]"
    return addName

def post_eth_sma(investment_value):
    eth_ret = pd.DataFrame(np.diff(ethsma['Close'])).rename(columns = {0:'returns'})
    sma_strategy_ret = []

    for i in range(len(eth_ret)):
        try:
            returns = eth_ret['returns'][i]*strategy['sma_position'][i]
            sma_strategy_ret.append(returns)
        except:
            pass
        
    sma_strategy_ret_df = pd.DataFrame(sma_strategy_ret).rename(columns = {0:'sma_returns'})

    number_of_stocks = investment_value/ethsma['Close'][1]
    sma_investment_ret = []
    #print(number_of_stocks)
    # print(ethsma['Close'][1])
    for i in range(len(sma_strategy_ret_df['sma_returns'])):
        returns = number_of_stocks*sma_strategy_ret_df['sma_returns'][i]
        sma_investment_ret.append(returns)

    sma_investment_ret_df = pd.DataFrame(sma_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(sma_investment_ret_df['investment_returns']), 2)
    
    return str(total_investment_ret)

#MACD
ethmacd = pd.read_csv('eth.csv').set_index('Date')
ethmacd.index =pd.to_datetime(ethmacd.index)

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'MACD'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'MACD':'Signal line'})
    hist = pd.DataFrame(macd['MACD'] - signal['Signal line']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

eth_macd = get_macd(ethmacd['Close'], 26, 12, 9)

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
            
buy_price, sell_price, macd_signal = implement_macd_strategy(ethmacd['Close'],eth_macd)

position = []
for i in range(len(macd_signal)):
    if macd_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(ethmacd['Close'])):
    if  macd_signal[i] == 1:
        position[i] = 1
    elif macd_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
macd = eth_macd['MACD']
signal = eth_macd['Signal line']
close_price = ethmacd['Close']
macd_signal = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(ethmacd.index)
position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(ethmacd.index)

macdframes = [close_price, macd, signal, macd_signal, position]
macdstrategy = pd.concat(macdframes, join = 'inner', axis = 1)
#print(macdstrategy)

ethmacd = pd.merge(ethmacd,macdstrategy[['macd_signal','macd_position']],how='outer' ,on=ethmacd.index)
ethmacd = ethmacd.rename(columns={'key_0':'Date'})
ethmacd = pd.merge(ethmacd,eth_macd[['MACD','Signal line','hist']],how='outer' ,on='Date')



def plot_eth_macd():
    ethmacd.set_index('Date',inplace=True)
    print(ethmacd)
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(ethmacd['Close'], color = 'skyblue', linewidth = 2, label = 'eth')
    ax1.plot(ethmacd.index, buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
    ax1.plot(ethmacd.index, sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
    ax1.legend()
    ax1.set_title('eth MACD SIGNALS')
    ax2.plot(eth_macd['MACD'], color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(eth_macd['Signal line'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(eth_macd)):
        if str(eth_macd['hist'][i])[0] == '-':
            ax2.bar(eth_macd.index[i], eth_macd['hist'][i], color = '#ef5350')
        else:
            ax2.bar(eth_macd.index[i], eth_macd['hist'][i], color = '#26a69a')
            
    plt.legend(loc = 'lower right')
    # plt.show()
    return(plt.savefig("eth_macd_output.jpg"))

def get_eth_macd():
    json_macd =  ethmacd.astype(str).to_json(orient='index')
    ethmacdName = "["+json_macd+","+"{\"name\": \"ETH\"}]"
    return ethmacdName

def post_eth_macd(investment_value):
    ethmacd_ret = pd.DataFrame(np.diff(ethmacd['Close'])).rename(columns = {0:'returns'})
    macd_strategy_ret = []

    for i in range(len(ethmacd_ret)):
        try:
            returns = ethmacd_ret['returns'][i]*macdstrategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass
        
    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns = {0:'macd_returns'})

    #investment_value = 100000
    number_of_stocks = floor(investment_value/ethmacd['Close'][0])
    macd_investment_ret = []

    for i in range(len(macd_strategy_ret_df['macd_returns'])):
        returns = number_of_stocks*macd_strategy_ret_df['macd_returns'][i]
        macd_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    #profit_percentage = floor((total_investment_ret/investment_value)*100)
    return(f'{total_investment_ret}')


## RSI
#if set_index date cant plot candle stick
ethrsi = pd.read_csv('eth.csv',header=0).set_index(['Date'])

ethrsi['diff'] = ethrsi['Close'].diff(1)

# Calculate Avg. Gains/Losses
ethrsi['gain'] = ethrsi['diff'].clip(lower=0).round(2)
ethrsi['loss'] = ethrsi['diff'].clip(upper=0).abs().round(2)

# Get initial Averages
window_length: int = 14
ethrsi['avg_gain'] = ethrsi['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
ethrsi['avg_loss'] = ethrsi['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]

# Get WMS averages
# Average Gains
for i, row in enumerate(ethrsi['avg_gain'].iloc[window_length+1:]):
    ethrsi['avg_gain'].iloc[i + window_length + 1] =\
        (ethrsi['avg_gain'].iloc[i + window_length] *
         (window_length - 1) +
         ethrsi['gain'].iloc[i + window_length + 1])\
        / window_length
# Average Losses
for i, row in enumerate(ethrsi['avg_loss'].iloc[window_length+1:]):
    ethrsi['avg_loss'].iloc[i + window_length + 1] =\
        (ethrsi['avg_loss'].iloc[i + window_length] *
         (window_length - 1) +
         ethrsi['loss'].iloc[i + window_length + 1])\
        / window_length
# View initial results

# Calculate RS Values
ethrsi['rs'] = ethrsi['avg_gain'] / ethrsi['avg_loss']

# Calculate RSI
ethrsi['rsi'] = 100 - (100 / (1.0 + ethrsi['rs']))

rsi = ethrsi['rsi']

def implement_rsi_strategy(prices, rsi):    
    buy_price = []
    sell_price = []
    rsi_signal = []
    signal = 0

    for i in range(len(rsi)):
        if rsi[i-1] > 30 and rsi[i] < 30:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
        elif rsi[i-1] < 70 and rsi[i] > 70:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            rsi_signal.append(0)
            
    return buy_price, sell_price, rsi_signal
            

buy_price, sell_price, rsi_signal = implement_rsi_strategy(ethrsi['Close'], ethrsi['rsi'])

position = []
for i in range(len(rsi_signal)):
    if rsi_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(ethrsi['Close'])):
    if rsi_signal[i] == 1:
        position[i] = 1
    elif rsi_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
rsi = ethrsi['rsi']
close_price = ethrsi['Close']
rsi_signal = pd.DataFrame(rsi_signal).rename(columns = {0:'rsi_signal'}).set_index(ethrsi.index)
position = pd.DataFrame(position).rename(columns = {0:'rsi_position'}).set_index(ethrsi.index)

rsiframes = [close_price, rsi, rsi_signal, position]
rsistrategy = pd.concat(rsiframes, join = 'inner', axis = 1)

rsistrategy.head()
# The value of the position remains 1 if we hold the stock or remains 0 if we sold or don’t own the stock. Finally, we are doing some data manipulations to combine all the created lists into one dataframe.
# From the output being shown, we can see that in the first row our position in the stock has remained 1 (since there isn’t any change in the RSI signal) but our position suddenly turned to 0 as we sold the stock when the Relative Strength Index trading signal represents a sell signal (-1). Our position will remain 0 until some changes in the trading signal occur.


def plot_eth_rsi():
    ethrsi.index = pd.to_datetime(ethrsi.index)
    plt.figure(figsize=(12,15))
    ax1 = plt.subplot2grid((5,1), (0,0), rowspan = 3, colspan = 1)
    ax2 = plt.subplot2grid((5,1), (3,0), rowspan = 2, colspan = 1)
    ax1.plot(ethrsi['Close'], linewidth = 2.5, color = 'skyblue', label = 'ETH')
    ax1.plot(ethrsi.index, buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
    ax1.plot(ethrsi.index, sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
    ax1.set_title('ETH RSI TRADE SIGNALS')
    ax2.plot(ethrsi['rsi'], color = 'orange', linewidth = 2.5)
    ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
    ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
    # plt.show()
    return(plt.savefig("eth_rsi_output.jpg"))


def post_eth_rsi(investment_value):
    ethrsi_ret = pd.DataFrame(np.diff(ethrsi['Close'])).rename(columns = {0:'returns'})
    rsi_strategy_ret = []

    for i in range(len(ethrsi_ret)):
        try:
            returns = ethrsi_ret['returns'][i]*rsistrategy['rsi_position'][i]
            rsi_strategy_ret.append(returns)
        except:
            pass
        
    rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns = {0:'rsi_returns'})

    
    number_of_stocks = investment_value/ethrsi['Close'][0]
    rsi_investment_ret = []

    for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
        returns = number_of_stocks*rsi_strategy_ret_df['rsi_returns'][i]
        rsi_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(rsi_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the RSI strategy by investing $100k in eth : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))
    return(f'{total_investment_ret}')


### Bolinger Bands
ethbb = pd.read_csv('eth.csv').set_index('Date')
ethbb.index =pd.to_datetime(ethbb.index)
def sma(data, window):
    sma = data.rolling(window = window).mean()
    return sma

ethbb['sma_20'] = sma(ethbb['Close'], 20)
ethbb.tail()

def bb(data, sma, window):
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

ethbb['upper_bb'], ethbb['lower_bb'] = bb(ethbb['Close'], ethbb['sma_20'], 20)
ethbb.tail()

def implement_bb_strategy(data, lower_bb, upper_bb):
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0
    
    for i in range(len(data)):
        if data[i-1] > lower_bb[i-1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        elif data[i-1] < upper_bb[i-1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)
            
    return buy_price, sell_price, bb_signal

buy_price, sell_price, bb_signal = implement_bb_strategy(ethbb['Close'], ethbb['lower_bb'], ethbb['upper_bb'])

def plot_eth_bb():
    ethbb['Close'].plot(label = 'CLOSE PRICES', alpha = 0.3)
    ethbb['upper_bb'].plot(label = 'UPPER BB', linestyle = '--', linewidth = 1, color = 'black')
    ##ethbb['middle_bb'].plot(label = 'MIDDLE BB', linestyle = '--', linewidth = 1.2, color = 'grey')
    ethbb['lower_bb'].plot(label = 'LOWER BB', linestyle = '--', linewidth = 1, color = 'black')
    plt.scatter(ethbb.index, buy_price, marker = '^', color = 'green', label = 'BUY', s = 200)
    plt.scatter(ethbb.index, sell_price, marker = 'v', color = 'red', label = 'SELL', s = 200)
    plt.title('eth BB STRATEGY TRADING SIGNALS')
    plt.legend(loc = 'upper left')
    return(plt.savefig("eth_bb_output.jpg"))


position = []
for i in range(len(bb_signal)):
    if bb_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(ethbb['Close'])):
    if bb_signal[i] == 1:
        position[i] = 1
    elif bb_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
upper_bb = ethbb['upper_bb']
lower_bb = ethbb['lower_bb']
close_price = ethbb['Close']
bb_signal = pd.DataFrame(bb_signal).rename(columns = {0:'bb_signal'}).set_index(ethbb.index)
position = pd.DataFrame(position).rename(columns = {0:'bb_position'}).set_index(ethbb.index)

bbframes = [close_price, upper_bb, lower_bb, bb_signal, position]
bbstrategy = pd.concat(bbframes, join = 'inner', axis = 1)
bbstrategy = bbstrategy.reset_index().drop('Date', axis = 1)

def post_eth_bb(investment_value):
    ethbb_ret = pd.DataFrame(np.diff(ethbb['Close'])).rename(columns = {0:'returns'})
    bb_strategy_ret = []

    for i in range(len(ethbb_ret)):
        try:
            returns =ethbb_ret['returns'][i]*bbstrategy['bb_position'][i]
            bb_strategy_ret.append(returns)
        except:
            pass
        
    bb_strategy_ret_df = pd.DataFrame(bb_strategy_ret).rename(columns = {0:'bb_returns'})

    number_of_stocks = investment_value/ethbb['Close'][-1]
    bb_investment_ret = []

    for i in range(len(bb_strategy_ret_df['bb_returns'])):
        returns = number_of_stocks*bb_strategy_ret_df['bb_returns'][i]
        bb_investment_ret.append(returns)

    bb_investment_ret_df = pd.DataFrame(bb_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_investment_ret_df['investment_returns']), 2)
    profit_percentage = math.floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the BB strategy by investing $100k in eth : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the BB strategy : {}%'.format(profit_percentage), attrs = ['bold']))
    return(f'{total_investment_ret}')