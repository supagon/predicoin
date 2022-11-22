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

paxg = web.get_data_yahoo('PAXG-USD', start='2021-01-01', end=datetime.now())

paxg.to_csv("paxg.csv")
paxg = pd.read_csv("paxg.csv")

def max_drawdown(paxg, window = 365):
    Roll_Max = paxg['Close'].rolling(window, min_periods=1).max()
    Daily_Drawdown = paxg['Close']/Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

    return Max_Daily_Drawdown

def get_paxg_info():
    paxginfo = pd.read_csv('paxg.csv')
    paxgdaily = pd.read_csv('paxg.csv',header=0)
    paxgdaily['Date'] = pd.to_datetime(paxgdaily['Date'])
    paxgdaily = paxgdaily.resample('D', on='Date').sum()
    paxgdaily['24hchange'] = paxgdaily['Adj Close'].pct_change()
    paxginfo = pd.merge(paxginfo,paxgdaily[['24hchange']],how='outer' ,on=paxginfo['Date'])
    paxginfo = paxginfo.drop(columns={'key_0'})
    paxginfo['Max_dd'] = max_drawdown(pd.read_csv('paxg.csv'))
    # print(paxginfo)
    json_result = paxginfo.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"PAXG\"}]"
    return addName

paxgsma = pd.read_csv('paxg.csv').set_index('Date')
paxgsma.index =pd.to_datetime(paxgsma.index)

def paxg_sma(data, n):
    sma = data.rolling(window = n).mean()
    return pd.DataFrame(sma)

n = [20, 50]
for i in n:
    paxgsma[f'sma_{i}'] = paxg_sma(paxgsma['Close'], i)
    
def paxg_implement_sma_strategy(data, short_window, long_window):
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


sma_20 = paxgsma['sma_20']
sma_50 = paxgsma['sma_50']

buy_price, sell_price, signal = paxg_implement_sma_strategy(paxgsma['Close'], sma_20, sma_50)

position = []
for i in range(len(signal)):
    if signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(paxgsma['Close'])):
    if signal[i] == 1:
        position[i] = 1
    elif signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]

sma_20 = pd.DataFrame(sma_20).rename(columns = {0:'sma_20'})
sma_50 = pd.DataFrame(sma_50).rename(columns = {0:'sma_50'}) 
signal = pd.DataFrame(signal).rename(columns = {0:'sma_signal'}).set_index(paxgsma.index)
position = pd.DataFrame(position).rename(columns = {0:'sma_position'}).set_index(paxgsma.index)

frames = [sma_20, sma_50, signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)
strategy = strategy.reset_index().drop('Date', axis = 1)



paxgsma = pd.merge(paxgsma,strategy[['sma_signal','sma_position']],how='outer' ,on=paxgsma.index)
paxgsma = paxgsma.rename(columns={'key_0':'Date'})

def plt_paxg_sma():
    paxgsma.set_index('Date',inplace=True)
    sma_20 = paxgsma['sma_20']
    sma_50 = paxgsma['sma_50']
    buy_price, sell_price, signal = paxg_implement_sma_strategy(paxgsma['Close'], sma_20, sma_50)
    plt.plot(paxgsma['Close'], alpha = 0.3, label = 'paxg')
    plt.plot(sma_20, alpha = 0.6, label = 'SMA 20')
    plt.plot(sma_50, alpha = 0.6, label = 'SMA 50')
    plt.scatter(paxgsma.index, buy_price, marker = '^', s = 200, color = 'darkblue', label = 'BUY SIGNAL')
    plt.scatter(paxgsma.index, sell_price, marker = 'v', s = 200, color = 'crimson', label = 'SELL SIGNAL')
    plt.legend(loc = 'upper left')
    plt.title('paxg SMA CROSSOVER TRADING SIGNALS')
    
    #date_form = DateFormatter("%d-%m-%Y")
    #plt.xaxis.set_major_formatter(date_form)
    #plt.show()
    #plt.savefig("output.jpg")
    return(plt.savefig("paxg_sma_output.jpg"))

def get_paxg_sma():
    json_result =  paxgsma.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"paxg\"}]"
    return addName

def post_paxg_sma(investment_value):
    paxg_ret = pd.DataFrame(np.diff(paxgsma['Close'])).rename(columns = {0:'returns'})
    sma_strategy_ret = []

    for i in range(len(paxg_ret)):
        try:
            returns = paxg_ret['returns'][i]*strategy['sma_position'][i]
            sma_strategy_ret.append(returns)
        except:
            pass
        
    sma_strategy_ret_df = pd.DataFrame(sma_strategy_ret).rename(columns = {0:'sma_returns'})

    number_of_stocks = investment_value/paxgsma['Close'][1]
    sma_investment_ret = []
    #print(number_of_stocks)
    # print(paxgsma['Close'][1])
    for i in range(len(sma_strategy_ret_df['sma_returns'])):
        returns = number_of_stocks*sma_strategy_ret_df['sma_returns'][i]
        sma_investment_ret.append(returns)

    sma_investment_ret_df = pd.DataFrame(sma_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(sma_investment_ret_df['investment_returns']), 2)
    
    return str(total_investment_ret)

#MACD
paxgmacd = pd.read_csv('paxg.csv').set_index('Date')
paxgmacd.index =pd.to_datetime(paxgmacd.index)

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'MACD'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'MACD':'Signal line'})
    hist = pd.DataFrame(macd['MACD'] - signal['Signal line']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

paxg_macd = get_macd(paxgmacd['Close'], 26, 12, 9)

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
            
buy_price, sell_price, macd_signal = implement_macd_strategy(paxgmacd['Close'],paxg_macd)

position = []
for i in range(len(macd_signal)):
    if macd_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(paxgmacd['Close'])):
    if  macd_signal[i] == 1:
        position[i] = 1
    elif macd_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
macd = paxg_macd['MACD']
signal = paxg_macd['Signal line']
close_price = paxgmacd['Close']
macd_signal = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(paxgmacd.index)
position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(paxgmacd.index)

macdframes = [close_price, macd, signal, macd_signal, position]
macdstrategy = pd.concat(macdframes, join = 'inner', axis = 1)
#print(macdstrategy)

paxgmacd = pd.merge(paxgmacd,macdstrategy[['macd_signal','macd_position']],how='outer' ,on=paxgmacd.index)
paxgmacd = paxgmacd.rename(columns={'key_0':'Date'})
paxgmacd = pd.merge(paxgmacd,paxg_macd[['MACD','Signal line','hist']],how='outer' ,on='Date')



def plot_paxg_macd():
    paxgmacd.set_index('Date',inplace=True)
    print(paxgmacd)
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(paxgmacd['Close'], color = 'skyblue', linewidth = 2, label = 'paxg')
    ax1.plot(paxgmacd.index, buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
    ax1.plot(paxgmacd.index, sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
    ax1.legend()
    ax1.set_title('paxg MACD SIGNALS')
    ax2.plot(paxg_macd['MACD'], color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(paxg_macd['Signal line'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(paxg_macd)):
        if str(paxg_macd['hist'][i])[0] == '-':
            ax2.bar(paxg_macd.index[i], paxg_macd['hist'][i], color = '#ef5350')
        else:
            ax2.bar(paxg_macd.index[i], paxg_macd['hist'][i], color = '#26a69a')
            
    plt.legend(loc = 'lower right')
    # plt.show()
    return(plt.savefig("paxg_macd_output.jpg"))

def get_paxg_macd():
    json_macd =  paxgmacd.astype(str).to_json(orient='index')
    paxgmacdName = "["+json_macd+","+"{\"name\": \"paxg\"}]"
    return paxgmacdName

def post_paxg_macd(investment_value):
    paxgmacd_ret = pd.DataFrame(np.diff(paxgmacd['Close'])).rename(columns = {0:'returns'})
    macd_strategy_ret = []

    for i in range(len(paxgmacd_ret)):
        try:
            returns = paxgmacd_ret['returns'][i]*macdstrategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass
        
    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns = {0:'macd_returns'})

    #investment_value = 100000
    number_of_stocks = floor(investment_value/paxgmacd['Close'][0])
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
paxgrsi = pd.read_csv('paxg.csv',header=0).set_index(['Date'])

paxgrsi['diff'] = paxgrsi['Close'].diff(1)

# Calculate Avg. Gains/Losses
paxgrsi['gain'] = paxgrsi['diff'].clip(lower=0).round(2)
paxgrsi['loss'] = paxgrsi['diff'].clip(upper=0).abs().round(2)

# Get initial Averages
window_length: int = 14
paxgrsi['avg_gain'] = paxgrsi['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
paxgrsi['avg_loss'] = paxgrsi['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]

# Get WMS averages
# Average Gains
for i, row in enumerate(paxgrsi['avg_gain'].iloc[window_length+1:]):
    paxgrsi['avg_gain'].iloc[i + window_length + 1] =\
        (paxgrsi['avg_gain'].iloc[i + window_length] *
         (window_length - 1) +
         paxgrsi['gain'].iloc[i + window_length + 1])\
        / window_length
# Average Losses
for i, row in enumerate(paxgrsi['avg_loss'].iloc[window_length+1:]):
    paxgrsi['avg_loss'].iloc[i + window_length + 1] =\
        (paxgrsi['avg_loss'].iloc[i + window_length] *
         (window_length - 1) +
         paxgrsi['loss'].iloc[i + window_length + 1])\
        / window_length
# View initial results

# Calculate RS Values
paxgrsi['rs'] = paxgrsi['avg_gain'] / paxgrsi['avg_loss']

# Calculate RSI
paxgrsi['rsi'] = 100 - (100 / (1.0 + paxgrsi['rs']))

rsi = paxgrsi['rsi']

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
            

buy_price, sell_price, rsi_signal = implement_rsi_strategy(paxgrsi['Close'], paxgrsi['rsi'])

position = []
for i in range(len(rsi_signal)):
    if rsi_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(paxgrsi['Close'])):
    if rsi_signal[i] == 1:
        position[i] = 1
    elif rsi_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
rsi = paxgrsi['rsi']
close_price = paxgrsi['Close']
rsi_signal = pd.DataFrame(rsi_signal).rename(columns = {0:'rsi_signal'}).set_index(paxgrsi.index)
position = pd.DataFrame(position).rename(columns = {0:'rsi_position'}).set_index(paxgrsi.index)

rsiframes = [close_price, rsi, rsi_signal, position]
rsistrategy = pd.concat(rsiframes, join = 'inner', axis = 1)

rsistrategy.head()
# The value of the position remains 1 if we hold the stock or remains 0 if we sold or don’t own the stock. Finally, we are doing some data manipulations to combine all the created lists into one dataframe.
# From the output being shown, we can see that in the first row our position in the stock has remained 1 (since there isn’t any change in the RSI signal) but our position suddenly turned to 0 as we sold the stock when the Relative Strength Index trading signal represents a sell signal (-1). Our position will remain 0 until some changes in the trading signal occur.


def plot_paxg_rsi():
    paxgrsi.index = pd.to_datetime(paxgrsi.index)
    plt.figure(figsize=(12,15))
    ax1 = plt.subplot2grid((5,1), (0,0), rowspan = 3, colspan = 1)
    ax2 = plt.subplot2grid((5,1), (3,0), rowspan = 2, colspan = 1)
    ax1.plot(paxgrsi['Close'], linewidth = 2.5, color = 'skyblue', label = 'PAXG')
    ax1.plot(paxgrsi.index, buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
    ax1.plot(paxgrsi.index, sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
    ax1.set_title('PAXG RSI TRADE SIGNALS')
    ax2.plot(paxgrsi['rsi'], color = 'orange', linewidth = 2.5)
    ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
    ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
    # plt.show()
    return(plt.savefig("paxg_rsi_output.jpg"))


def post_paxg_rsi(investment_value):
    paxgrsi_ret = pd.DataFrame(np.diff(paxgrsi['Close'])).rename(columns = {0:'returns'})
    rsi_strategy_ret = []

    for i in range(len(paxgrsi_ret)):
        try:
            returns = paxgrsi_ret['returns'][i]*rsistrategy['rsi_position'][i]
            rsi_strategy_ret.append(returns)
        except:
            pass
        
    rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns = {0:'rsi_returns'})

    
    number_of_stocks = investment_value/paxgrsi['Close'][0]
    rsi_investment_ret = []

    for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
        returns = number_of_stocks*rsi_strategy_ret_df['rsi_returns'][i]
        rsi_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(rsi_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the RSI strategy by investing $100k in paxg : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))
    return(f'{total_investment_ret}')


### Bolinger Bands
paxgbb = pd.read_csv('paxg.csv').set_index('Date')
paxgbb.index =pd.to_datetime(paxgbb.index)
def sma(data, window):
    sma = data.rolling(window = window).mean()
    return sma

paxgbb['sma_20'] = sma(paxgbb['Close'], 20)
paxgbb.tail()

def bb(data, sma, window):
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

paxgbb['upper_bb'], paxgbb['lower_bb'] = bb(paxgbb['Close'], paxgbb['sma_20'], 20)
paxgbb.tail()

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

buy_price, sell_price, bb_signal = implement_bb_strategy(paxgbb['Close'], paxgbb['lower_bb'], paxgbb['upper_bb'])

def plot_paxg_bb():
    paxgbb['Close'].plot(label = 'CLOSE PRICES', alpha = 0.3)
    paxgbb['upper_bb'].plot(label = 'UPPER BB', linestyle = '--', linewidth = 1, color = 'black')
    ##paxgbb['middle_bb'].plot(label = 'MIDDLE BB', linestyle = '--', linewidth = 1.2, color = 'grey')
    paxgbb['lower_bb'].plot(label = 'LOWER BB', linestyle = '--', linewidth = 1, color = 'black')
    plt.scatter(paxgbb.index, buy_price, marker = '^', color = 'green', label = 'BUY', s = 200)
    plt.scatter(paxgbb.index, sell_price, marker = 'v', color = 'red', label = 'SELL', s = 200)
    plt.title('paxg BB STRATEGY TRADING SIGNALS')
    plt.legend(loc = 'upper left')
    return(plt.savefig("paxg_bb_output.jpg"))


position = []
for i in range(len(bb_signal)):
    if bb_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(paxgbb['Close'])):
    if bb_signal[i] == 1:
        position[i] = 1
    elif bb_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
upper_bb = paxgbb['upper_bb']
lower_bb = paxgbb['lower_bb']
close_price = paxgbb['Close']
bb_signal = pd.DataFrame(bb_signal).rename(columns = {0:'bb_signal'}).set_index(paxgbb.index)
position = pd.DataFrame(position).rename(columns = {0:'bb_position'}).set_index(paxgbb.index)

bbframes = [close_price, upper_bb, lower_bb, bb_signal, position]
bbstrategy = pd.concat(bbframes, join = 'inner', axis = 1)
bbstrategy = bbstrategy.reset_index().drop('Date', axis = 1)

def post_paxg_bb(investment_value):
    paxgbb_ret = pd.DataFrame(np.diff(paxgbb['Close'])).rename(columns = {0:'returns'})
    bb_strategy_ret = []

    for i in range(len(paxgbb_ret)):
        try:
            returns =paxgbb_ret['returns'][i]*bbstrategy['bb_position'][i]
            bb_strategy_ret.append(returns)
        except:
            pass
        
    bb_strategy_ret_df = pd.DataFrame(bb_strategy_ret).rename(columns = {0:'bb_returns'})

    number_of_stocks = investment_value/paxgbb['Close'][-1]
    bb_investment_ret = []

    for i in range(len(bb_strategy_ret_df['bb_returns'])):
        returns = number_of_stocks*bb_strategy_ret_df['bb_returns'][i]
        bb_investment_ret.append(returns)

    bb_investment_ret_df = pd.DataFrame(bb_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_investment_ret_df['investment_returns']), 2)
    profit_percentage = math.floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the BB strategy by investing $100k in paxg : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the BB strategy : {}%'.format(profit_percentage), attrs = ['bold']))
    return(f'{total_investment_ret}')