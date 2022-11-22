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

def max_drawdown(yfi, window = 365):
    Roll_Max = yfi['Close'].rolling(window, min_periods=1).max()
    Daily_Drawdown = yfi['Close']/Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

    return Max_Daily_Drawdown

def get_yfi_info():
    yfiinfo = pd.read_csv('yfi.csv')
    yfidaily = pd.read_csv('yfi.csv',header=0)
    yfidaily['Date'] = pd.to_datetime(yfidaily['Date'])
    yfidaily = yfidaily.resample('D', on='Date').sum()
    yfidaily['24hchange'] = yfidaily['Adj Close'].pct_change()
    yfiinfo = pd.merge(yfiinfo,yfidaily[['24hchange']],how='outer' ,on=yfiinfo['Date'])
    yfiinfo = yfiinfo.drop(columns={'key_0'})
    yfiinfo['Max_dd'] = max_drawdown(pd.read_csv('yfi.csv'))
    #print(yfiinfo)
    json_result = yfiinfo.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"YFI\"}]"
    return addName

yfisma = pd.read_csv('yfi.csv').set_index('Date')
yfisma.index =pd.to_datetime(yfisma.index)

def yfi_sma(data, n):
    sma = data.rolling(window = n).mean()
    return pd.DataFrame(sma)

n = [20, 50]
for i in n:
    yfisma[f'sma_{i}'] = yfi_sma(yfisma['Close'], i)
    
def yfi_implement_sma_strategy(data, short_window, long_window):
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


sma_20 = yfisma['sma_20']
sma_50 = yfisma['sma_50']

buy_price, sell_price, signal = yfi_implement_sma_strategy(yfisma['Close'], sma_20, sma_50)

position = []
for i in range(len(signal)):
    if signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(yfisma['Close'])):
    if signal[i] == 1:
        position[i] = 1
    elif signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]

sma_20 = pd.DataFrame(sma_20).rename(columns = {0:'sma_20'})
sma_50 = pd.DataFrame(sma_50).rename(columns = {0:'sma_50'}) 
signal = pd.DataFrame(signal).rename(columns = {0:'sma_signal'}).set_index(yfisma.index)
position = pd.DataFrame(position).rename(columns = {0:'sma_position'}).set_index(yfisma.index)

frames = [sma_20, sma_50, signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)
strategy = strategy.reset_index().drop('Date', axis = 1)



yfisma = pd.merge(yfisma,strategy[['sma_signal','sma_position']],how='outer' ,on=yfisma.index)
yfisma = yfisma.rename(columns={'key_0':'Date'})

def plt_yfi_sma():
    yfisma.set_index('Date',inplace=True)
    sma_20 = yfisma['sma_20']
    sma_50 = yfisma['sma_50']
    buy_price, sell_price, signal = yfi_implement_sma_strategy(yfisma['Close'], sma_20, sma_50)
    plt.plot(yfisma['Close'], alpha = 0.3, label = 'YFI')
    plt.plot(sma_20, alpha = 0.6, label = 'SMA 20')
    plt.plot(sma_50, alpha = 0.6, label = 'SMA 50')
    plt.scatter(yfisma.index, buy_price, marker = '^', s = 200, color = 'darkblue', label = 'BUY SIGNAL')
    plt.scatter(yfisma.index, sell_price, marker = 'v', s = 200, color = 'crimson', label = 'SELL SIGNAL')
    plt.legend(loc = 'upper left')
    plt.title('YFI SMA CROSSOVER TRADING SIGNALS')
    
    #date_form = DateFormatter("%d-%m-%Y")
    #plt.xaxis.set_major_formatter(date_form)
    #plt.show()
    #plt.savefig("output.jpg")
    return(plt.savefig("yfi_sma_output.jpg"))

def get_yfi_sma():
    json_result =  yfisma.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"yfi\"}]"
    return addName

def post_yfi_sma(investment_value):
    yfi_ret = pd.DataFrame(np.diff(yfisma['Close'])).rename(columns = {0:'returns'})
    sma_strategy_ret = []

    for i in range(len(yfi_ret)):
        try:
            returns = yfi_ret['returns'][i]*strategy['sma_position'][i]
            sma_strategy_ret.append(returns)
        except:
            pass
        
    sma_strategy_ret_df = pd.DataFrame(sma_strategy_ret).rename(columns = {0:'sma_returns'})

    number_of_stocks = investment_value/yfisma['Close'][1]
    sma_investment_ret = []
    #print(number_of_stocks)
    # print(yfisma['Close'][1])
    for i in range(len(sma_strategy_ret_df['sma_returns'])):
        returns = number_of_stocks*sma_strategy_ret_df['sma_returns'][i]
        sma_investment_ret.append(returns)

    sma_investment_ret_df = pd.DataFrame(sma_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(sma_investment_ret_df['investment_returns']), 2)
    
    return str(total_investment_ret)

#MACD
yfimacd = pd.read_csv('yfi.csv').set_index('Date')
yfimacd.index =pd.to_datetime(yfimacd.index)

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'MACD'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'MACD':'Signal line'})
    hist = pd.DataFrame(macd['MACD'] - signal['Signal line']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

yfi_macd = get_macd(yfimacd['Close'], 26, 12, 9)

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
            
buy_price, sell_price, macd_signal = implement_macd_strategy(yfimacd['Close'],yfi_macd)

position = []
for i in range(len(macd_signal)):
    if macd_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(yfimacd['Close'])):
    if  macd_signal[i] == 1:
        position[i] = 1
    elif macd_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
macd = yfi_macd['MACD']
signal = yfi_macd['Signal line']
close_price = yfimacd['Close']
macd_signal = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(yfimacd.index)
position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(yfimacd.index)

macdframes = [close_price, macd, signal, macd_signal, position]
macdstrategy = pd.concat(macdframes, join = 'inner', axis = 1)
#print(macdstrategy)

yfimacd = pd.merge(yfimacd,macdstrategy[['macd_signal','macd_position']],how='outer' ,on=yfimacd.index)
yfimacd = yfimacd.rename(columns={'key_0':'Date'})
yfimacd = pd.merge(yfimacd,yfi_macd[['MACD','Signal line','hist']],how='outer' ,on='Date')



def plot_yfi_macd():
    yfimacd.set_index('Date',inplace=True)
    print(yfimacd)
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(yfimacd['Close'], color = 'skyblue', linewidth = 2, label = 'YFI')
    ax1.plot(yfimacd.index, buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
    ax1.plot(yfimacd.index, sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
    ax1.legend()
    ax1.set_title('YFI MACD SIGNALS')
    ax2.plot(yfi_macd['MACD'], color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(yfi_macd['Signal line'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(yfi_macd)):
        if str(yfi_macd['hist'][i])[0] == '-':
            ax2.bar(yfi_macd.index[i], yfi_macd['hist'][i], color = '#ef5350')
        else:
            ax2.bar(yfi_macd.index[i], yfi_macd['hist'][i], color = '#26a69a')
            
    plt.legend(loc = 'lower right')
    # plt.show()
    return(plt.savefig("yfi_macd_output.jpg"))

def get_yfi_macd():
    json_macd =  yfimacd.astype(str).to_json(orient='index')
    yfimacdName = "["+json_macd+","+"{\"name\": \"yfi\"}]"
    return yfimacdName

def post_yfi_macd(investment_value):
    yfimacd_ret = pd.DataFrame(np.diff(yfimacd['Close'])).rename(columns = {0:'returns'})
    macd_strategy_ret = []

    for i in range(len(yfimacd_ret)):
        try:
            returns = yfimacd_ret['returns'][i]*macdstrategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass
        
    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns = {0:'macd_returns'})

    #investment_value = 100000
    number_of_stocks = floor(investment_value/yfimacd['Close'][0])
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
yfirsi = pd.read_csv('yfi.csv',header=0).set_index(['Date'])

yfirsi['diff'] = yfirsi['Close'].diff(1)

# Calculate Avg. Gains/Losses
yfirsi['gain'] = yfirsi['diff'].clip(lower=0).round(2)
yfirsi['loss'] = yfirsi['diff'].clip(upper=0).abs().round(2)

# Get initial Averages
window_length: int = 14
yfirsi['avg_gain'] = yfirsi['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
yfirsi['avg_loss'] = yfirsi['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]

# Get WMS averages
# Average Gains
for i, row in enumerate(yfirsi['avg_gain'].iloc[window_length+1:]):
    yfirsi['avg_gain'].iloc[i + window_length + 1] =\
        (yfirsi['avg_gain'].iloc[i + window_length] *
         (window_length - 1) +
         yfirsi['gain'].iloc[i + window_length + 1])\
        / window_length
# Average Losses
for i, row in enumerate(yfirsi['avg_loss'].iloc[window_length+1:]):
    yfirsi['avg_loss'].iloc[i + window_length + 1] =\
        (yfirsi['avg_loss'].iloc[i + window_length] *
         (window_length - 1) +
         yfirsi['loss'].iloc[i + window_length + 1])\
        / window_length
# View initial results

# Calculate RS Values
yfirsi['rs'] = yfirsi['avg_gain'] / yfirsi['avg_loss']

# Calculate RSI
yfirsi['rsi'] = 100 - (100 / (1.0 + yfirsi['rs']))

rsi = yfirsi['rsi']

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
            

buy_price, sell_price, rsi_signal = implement_rsi_strategy(yfirsi['Close'], yfirsi['rsi'])

position = []
for i in range(len(rsi_signal)):
    if rsi_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(yfirsi['Close'])):
    if rsi_signal[i] == 1:
        position[i] = 1
    elif rsi_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
rsi = yfirsi['rsi']
close_price = yfirsi['Close']
rsi_signal = pd.DataFrame(rsi_signal).rename(columns = {0:'rsi_signal'}).set_index(yfirsi.index)
position = pd.DataFrame(position).rename(columns = {0:'rsi_position'}).set_index(yfirsi.index)

rsiframes = [close_price, rsi, rsi_signal, position]
rsistrategy = pd.concat(rsiframes, join = 'inner', axis = 1)

rsistrategy.head()
# The value of the position remains 1 if we hold the stock or remains 0 if we yfid or don’t own the stock. Finally, we are doing some data manipulations to combine all the created lists into one dataframe.
# From the output being shown, we can see that in the first row our position in the stock has remained 1 (since there isn’t any change in the RSI signal) but our position suddenly turned to 0 as we yfid the stock when the Relative Strength Index trading signal represents a sell signal (-1). Our position will remain 0 until some changes in the trading signal occur.


def plot_yfi_rsi():
    yfirsi.index = pd.to_datetime(yfirsi.index)
    plt.figure(figsize=(12,15))
    ax1 = plt.subplot2grid((5,1), (0,0), rowspan = 3, colspan = 1)
    ax2 = plt.subplot2grid((5,1), (3,0), rowspan = 2, colspan = 1)
    ax1.plot(yfirsi['Close'], linewidth = 2.5, color = 'skyblue', label = 'YFI')
    ax1.plot(yfirsi.index, buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
    ax1.plot(yfirsi.index, sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
    ax1.set_title('YFI RSI TRADE SIGNALS')
    ax2.plot(yfirsi['rsi'], color = 'orange', linewidth = 2.5)
    ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
    ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
    # plt.show()
    return(plt.savefig("yfi_rsi_output.jpg"))


def post_yfi_rsi(investment_value):
    yfirsi_ret = pd.DataFrame(np.diff(yfirsi['Close'])).rename(columns = {0:'returns'})
    rsi_strategy_ret = []

    for i in range(len(yfirsi_ret)):
        try:
            returns = yfirsi_ret['returns'][i]*rsistrategy['rsi_position'][i]
            rsi_strategy_ret.append(returns)
        except:
            pass
        
    rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns = {0:'rsi_returns'})

    
    number_of_stocks = investment_value/yfirsi['Close'][0]
    rsi_investment_ret = []

    for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
        returns = number_of_stocks*rsi_strategy_ret_df['rsi_returns'][i]
        rsi_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(rsi_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the RSI strategy by investing $100k in yfi : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))
    return(f'{total_investment_ret}')


### Bolinger Bands
yfibb = pd.read_csv('yfi.csv').set_index('Date')
yfibb.index =pd.to_datetime(yfibb.index)
def sma(data, window):
    sma = data.rolling(window = window).mean()
    return sma

yfibb['sma_20'] = sma(yfibb['Close'], 20)
yfibb.tail()

def bb(data, sma, window):
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

yfibb['upper_bb'], yfibb['lower_bb'] = bb(yfibb['Close'], yfibb['sma_20'], 20)
yfibb.tail()

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

buy_price, sell_price, bb_signal = implement_bb_strategy(yfibb['Close'], yfibb['lower_bb'], yfibb['upper_bb'])

def plot_yfi_bb():
    yfibb['Close'].plot(label = 'CLOSE PRICES', alpha = 0.3)
    yfibb['upper_bb'].plot(label = 'UPPER BB', linestyle = '--', linewidth = 1, color = 'black')
    ##yfibb['middle_bb'].plot(label = 'MIDDLE BB', linestyle = '--', linewidth = 1.2, color = 'grey')
    yfibb['lower_bb'].plot(label = 'LOWER BB', linestyle = '--', linewidth = 1, color = 'black')
    plt.scatter(yfibb.index, buy_price, marker = '^', color = 'green', label = 'BUY', s = 200)
    plt.scatter(yfibb.index, sell_price, marker = 'v', color = 'red', label = 'SELL', s = 200)
    plt.title('YFI BB STRATEGY TRADING SIGNALS')
    plt.legend(loc = 'upper left')
    return(plt.savefig("yfi_bb_output.jpg"))


position = []
for i in range(len(bb_signal)):
    if bb_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(yfibb['Close'])):
    if bb_signal[i] == 1:
        position[i] = 1
    elif bb_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
upper_bb = yfibb['upper_bb']
lower_bb = yfibb['lower_bb']
close_price = yfibb['Close']
bb_signal = pd.DataFrame(bb_signal).rename(columns = {0:'bb_signal'}).set_index(yfibb.index)
position = pd.DataFrame(position).rename(columns = {0:'bb_position'}).set_index(yfibb.index)

bbframes = [close_price, upper_bb, lower_bb, bb_signal, position]
bbstrategy = pd.concat(bbframes, join = 'inner', axis = 1)
bbstrategy = bbstrategy.reset_index().drop('Date', axis = 1)

def post_yfi_bb(investment_value):
    yfibb_ret = pd.DataFrame(np.diff(yfibb['Close'])).rename(columns = {0:'returns'})
    bb_strategy_ret = []

    for i in range(len(yfibb_ret)):
        try:
            returns =yfibb_ret['returns'][i]*bbstrategy['bb_position'][i]
            bb_strategy_ret.append(returns)
        except:
            pass
        
    bb_strategy_ret_df = pd.DataFrame(bb_strategy_ret).rename(columns = {0:'bb_returns'})

    number_of_stocks = investment_value/yfibb['Close'][-1]
    bb_investment_ret = []

    for i in range(len(bb_strategy_ret_df['bb_returns'])):
        returns = number_of_stocks*bb_strategy_ret_df['bb_returns'][i]
        bb_investment_ret.append(returns)

    bb_investment_ret_df = pd.DataFrame(bb_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_investment_ret_df['investment_returns']), 2)
    profit_percentage = math.floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the BB strategy by investing $100k in yfi : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the BB strategy : {}%'.format(profit_percentage), attrs = ['bold']))
    return(f'{total_investment_ret}')