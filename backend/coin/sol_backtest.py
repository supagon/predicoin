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

def max_drawdown(sol, window = 365):
    Roll_Max = sol['Close'].rolling(window, min_periods=1).max()
    Daily_Drawdown = sol['Close']/Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

    return Max_Daily_Drawdown

def get_sol_info():
    solinfo = pd.read_csv('sol.csv')
    soldaily = pd.read_csv('sol.csv',header=0)
    soldaily['Date'] = pd.to_datetime(soldaily['Date'])
    soldaily = soldaily.resample('D', on='Date').sum()
    soldaily['24hchange'] = soldaily['Adj Close'].pct_change()
    solinfo = pd.merge(solinfo,soldaily[['24hchange']],how='outer' ,on=solinfo['Date'])
    solinfo = solinfo.drop(columns={'key_0'})
    solinfo['Max_dd'] = max_drawdown(pd.read_csv('sol.csv'))
    #print(solinfo)
    json_result = solinfo.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"SOL\"}]"
    return addName

solsma = pd.read_csv('sol.csv').set_index('Date')
solsma.index =pd.to_datetime(solsma.index)

def sol_sma(data, n):
    sma = data.rolling(window = n).mean()
    return pd.DataFrame(sma)

n = [20, 50]
for i in n:
    solsma[f'sma_{i}'] = sol_sma(solsma['Close'], i)
    
def sol_implement_sma_strategy(data, short_window, long_window):
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


sma_20 = solsma['sma_20']
sma_50 = solsma['sma_50']

buy_price, sell_price, signal = sol_implement_sma_strategy(solsma['Close'], sma_20, sma_50)

position = []
for i in range(len(signal)):
    if signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(solsma['Close'])):
    if signal[i] == 1:
        position[i] = 1
    elif signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]

sma_20 = pd.DataFrame(sma_20).rename(columns = {0:'sma_20'})
sma_50 = pd.DataFrame(sma_50).rename(columns = {0:'sma_50'}) 
signal = pd.DataFrame(signal).rename(columns = {0:'sma_signal'}).set_index(solsma.index)
position = pd.DataFrame(position).rename(columns = {0:'sma_position'}).set_index(solsma.index)

frames = [sma_20, sma_50, signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)
strategy = strategy.reset_index().drop('Date', axis = 1)



solsma = pd.merge(solsma,strategy[['sma_signal','sma_position']],how='outer' ,on=solsma.index)
solsma = solsma.rename(columns={'key_0':'Date'})

def plt_sol_sma():
    solsma.set_index('Date',inplace=True)
    sma_20 = solsma['sma_20']
    sma_50 = solsma['sma_50']
    buy_price, sell_price, signal = sol_implement_sma_strategy(solsma['Close'], sma_20, sma_50)
    plt.plot(solsma['Close'], alpha = 0.3, label = 'SOL')
    plt.plot(sma_20, alpha = 0.6, label = 'SMA 20')
    plt.plot(sma_50, alpha = 0.6, label = 'SMA 50')
    plt.scatter(solsma.index, buy_price, marker = '^', s = 200, color = 'darkblue', label = 'BUY SIGNAL')
    plt.scatter(solsma.index, sell_price, marker = 'v', s = 200, color = 'crimson', label = 'SELL SIGNAL')
    plt.legend(loc = 'upper left')
    plt.title('SOL SMA CROSSOVER TRADING SIGNALS')
    
    #date_form = DateFormatter("%d-%m-%Y")
    #plt.xaxis.set_major_formatter(date_form)
    #plt.show()
    #plt.savefig("output.jpg")
    return(plt.savefig("sol_sma_output.jpg"))

def get_sol_sma():
    json_result =  solsma.astype(str).to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"sol\"}]"
    return addName

def post_sol_sma(investment_value):
    sol_ret = pd.DataFrame(np.diff(solsma['Close'])).rename(columns = {0:'returns'})
    sma_strategy_ret = []

    for i in range(len(sol_ret)):
        try:
            returns = sol_ret['returns'][i]*strategy['sma_position'][i]
            sma_strategy_ret.append(returns)
        except:
            pass
        
    sma_strategy_ret_df = pd.DataFrame(sma_strategy_ret).rename(columns = {0:'sma_returns'})

    number_of_stocks = investment_value/solsma['Close'][1]
    sma_investment_ret = []
    #print(number_of_stocks)
    # print(solsma['Close'][1])
    for i in range(len(sma_strategy_ret_df['sma_returns'])):
        returns = number_of_stocks*sma_strategy_ret_df['sma_returns'][i]
        sma_investment_ret.append(returns)

    sma_investment_ret_df = pd.DataFrame(sma_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(sma_investment_ret_df['investment_returns']), 2)
    
    return str(total_investment_ret)

#MACD
solmacd = pd.read_csv('sol.csv').set_index('Date')
solmacd.index =pd.to_datetime(solmacd.index)

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'MACD'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'MACD':'Signal line'})
    hist = pd.DataFrame(macd['MACD'] - signal['Signal line']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

sol_macd = get_macd(solmacd['Close'], 26, 12, 9)

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
            
buy_price, sell_price, macd_signal = implement_macd_strategy(solmacd['Close'],sol_macd)

position = []
for i in range(len(macd_signal)):
    if macd_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(solmacd['Close'])):
    if  macd_signal[i] == 1:
        position[i] = 1
    elif macd_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
macd = sol_macd['MACD']
signal = sol_macd['Signal line']
close_price = solmacd['Close']
macd_signal = pd.DataFrame(macd_signal).rename(columns = {0:'macd_signal'}).set_index(solmacd.index)
position = pd.DataFrame(position).rename(columns = {0:'macd_position'}).set_index(solmacd.index)

macdframes = [close_price, macd, signal, macd_signal, position]
macdstrategy = pd.concat(macdframes, join = 'inner', axis = 1)
#print(macdstrategy)

solmacd = pd.merge(solmacd,macdstrategy[['macd_signal','macd_position']],how='outer' ,on=solmacd.index)
solmacd = solmacd.rename(columns={'key_0':'Date'})
solmacd = pd.merge(solmacd,sol_macd[['MACD','Signal line','hist']],how='outer' ,on='Date')



def plot_sol_macd():
    solmacd.set_index('Date',inplace=True)
    print(solmacd)
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(solmacd['Close'], color = 'skyblue', linewidth = 2, label = 'SOL')
    ax1.plot(solmacd.index, buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
    ax1.plot(solmacd.index, sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
    ax1.legend()
    ax1.set_title('SOL MACD SIGNALS')
    ax2.plot(sol_macd['MACD'], color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(sol_macd['Signal line'], color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(sol_macd)):
        if str(sol_macd['hist'][i])[0] == '-':
            ax2.bar(sol_macd.index[i], sol_macd['hist'][i], color = '#ef5350')
        else:
            ax2.bar(sol_macd.index[i], sol_macd['hist'][i], color = '#26a69a')
            
    plt.legend(loc = 'lower right')
    # plt.show()
    return(plt.savefig("sol_macd_output.jpg"))

def get_sol_macd():
    json_macd =  solmacd.astype(str).to_json(orient='index')
    solmacdName = "["+json_macd+","+"{\"name\": \"sol\"}]"
    return solmacdName

def post_sol_macd(investment_value):
    solmacd_ret = pd.DataFrame(np.diff(solmacd['Close'])).rename(columns = {0:'returns'})
    macd_strategy_ret = []

    for i in range(len(solmacd_ret)):
        try:
            returns = solmacd_ret['returns'][i]*macdstrategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass
        
    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns = {0:'macd_returns'})

    #investment_value = 100000
    number_of_stocks = floor(investment_value/solmacd['Close'][0])
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
solrsi = pd.read_csv('sol.csv',header=0).set_index(['Date'])

solrsi['diff'] = solrsi['Close'].diff(1)

# Calculate Avg. Gains/Losses
solrsi['gain'] = solrsi['diff'].clip(lower=0).round(2)
solrsi['loss'] = solrsi['diff'].clip(upper=0).abs().round(2)

# Get initial Averages
window_length: int = 14
solrsi['avg_gain'] = solrsi['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
solrsi['avg_loss'] = solrsi['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]

# Get WMS averages
# Average Gains
for i, row in enumerate(solrsi['avg_gain'].iloc[window_length+1:]):
    solrsi['avg_gain'].iloc[i + window_length + 1] =\
        (solrsi['avg_gain'].iloc[i + window_length] *
         (window_length - 1) +
         solrsi['gain'].iloc[i + window_length + 1])\
        / window_length
# Average Losses
for i, row in enumerate(solrsi['avg_loss'].iloc[window_length+1:]):
    solrsi['avg_loss'].iloc[i + window_length + 1] =\
        (solrsi['avg_loss'].iloc[i + window_length] *
         (window_length - 1) +
         solrsi['loss'].iloc[i + window_length + 1])\
        / window_length
# View initial results

# Calculate RS Values
solrsi['rs'] = solrsi['avg_gain'] / solrsi['avg_loss']

# Calculate RSI
solrsi['rsi'] = 100 - (100 / (1.0 + solrsi['rs']))

rsi = solrsi['rsi']

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
            

buy_price, sell_price, rsi_signal = implement_rsi_strategy(solrsi['Close'], solrsi['rsi'])

position = []
for i in range(len(rsi_signal)):
    if rsi_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(solrsi['Close'])):
    if rsi_signal[i] == 1:
        position[i] = 1
    elif rsi_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
rsi = solrsi['rsi']
close_price = solrsi['Close']
rsi_signal = pd.DataFrame(rsi_signal).rename(columns = {0:'rsi_signal'}).set_index(solrsi.index)
position = pd.DataFrame(position).rename(columns = {0:'rsi_position'}).set_index(solrsi.index)

rsiframes = [close_price, rsi, rsi_signal, position]
rsistrategy = pd.concat(rsiframes, join = 'inner', axis = 1)

rsistrategy.head()
# The value of the position remains 1 if we hold the stock or remains 0 if we sold or don’t own the stock. Finally, we are doing some data manipulations to combine all the created lists into one dataframe.
# From the output being shown, we can see that in the first row our position in the stock has remained 1 (since there isn’t any change in the RSI signal) but our position suddenly turned to 0 as we sold the stock when the Relative Strength Index trading signal represents a sell signal (-1). Our position will remain 0 until some changes in the trading signal occur.


def plot_sol_rsi():
    solrsi.index = pd.to_datetime(solrsi.index)
    plt.figure(figsize=(12,15))
    ax1 = plt.subplot2grid((5,1), (0,0), rowspan = 3, colspan = 1)
    ax2 = plt.subplot2grid((5,1), (3,0), rowspan = 2, colspan = 1)
    ax1.plot(solrsi['Close'], linewidth = 2.5, color = 'skyblue', label = 'SOL')
    ax1.plot(solrsi.index, buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
    ax1.plot(solrsi.index, sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
    ax1.set_title('SOL RSI TRADE SIGNALS')
    ax2.plot(solrsi['rsi'], color = 'orange', linewidth = 2.5)
    ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
    ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
    # plt.show()
    return(plt.savefig("sol_rsi_output.jpg"))


def post_sol_rsi(investment_value):
    solrsi_ret = pd.DataFrame(np.diff(solrsi['Close'])).rename(columns = {0:'returns'})
    rsi_strategy_ret = []

    for i in range(len(solrsi_ret)):
        try:
            returns = solrsi_ret['returns'][i]*rsistrategy['rsi_position'][i]
            rsi_strategy_ret.append(returns)
        except:
            pass
        
    rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns = {0:'rsi_returns'})

    
    number_of_stocks = investment_value/solrsi['Close'][0]
    rsi_investment_ret = []

    for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
        returns = number_of_stocks*rsi_strategy_ret_df['rsi_returns'][i]
        rsi_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(rsi_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the RSI strategy by investing $100k in sol : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))
    return(f'{total_investment_ret}')


### Bolinger Bands
solbb = pd.read_csv('sol.csv').set_index('Date')
solbb.index =pd.to_datetime(solbb.index)
def sma(data, window):
    sma = data.rolling(window = window).mean()
    return sma

solbb['sma_20'] = sma(solbb['Close'], 20)
solbb.tail()

def bb(data, sma, window):
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

solbb['upper_bb'], solbb['lower_bb'] = bb(solbb['Close'], solbb['sma_20'], 20)
solbb.tail()

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

buy_price, sell_price, bb_signal = implement_bb_strategy(solbb['Close'], solbb['lower_bb'], solbb['upper_bb'])

def plot_sol_bb():
    solbb['Close'].plot(label = 'CLOSE PRICES', alpha = 0.3)
    solbb['upper_bb'].plot(label = 'UPPER BB', linestyle = '--', linewidth = 1, color = 'black')
    ##solbb['middle_bb'].plot(label = 'MIDDLE BB', linestyle = '--', linewidth = 1.2, color = 'grey')
    solbb['lower_bb'].plot(label = 'LOWER BB', linestyle = '--', linewidth = 1, color = 'black')
    plt.scatter(solbb.index, buy_price, marker = '^', color = 'green', label = 'BUY', s = 200)
    plt.scatter(solbb.index, sell_price, marker = 'v', color = 'red', label = 'SELL', s = 200)
    plt.title('SOL BB STRATEGY TRADING SIGNALS')
    plt.legend(loc = 'upper left')
    return(plt.savefig("sol_bb_output.jpg"))


position = []
for i in range(len(bb_signal)):
    if bb_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(solbb['Close'])):
    if bb_signal[i] == 1:
        position[i] = 1
    elif bb_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
upper_bb = solbb['upper_bb']
lower_bb = solbb['lower_bb']
close_price = solbb['Close']
bb_signal = pd.DataFrame(bb_signal).rename(columns = {0:'bb_signal'}).set_index(solbb.index)
position = pd.DataFrame(position).rename(columns = {0:'bb_position'}).set_index(solbb.index)

bbframes = [close_price, upper_bb, lower_bb, bb_signal, position]
bbstrategy = pd.concat(bbframes, join = 'inner', axis = 1)
bbstrategy = bbstrategy.reset_index().drop('Date', axis = 1)

def post_sol_bb(investment_value):
    solbb_ret = pd.DataFrame(np.diff(solbb['Close'])).rename(columns = {0:'returns'})
    bb_strategy_ret = []

    for i in range(len(solbb_ret)):
        try:
            returns =solbb_ret['returns'][i]*bbstrategy['bb_position'][i]
            bb_strategy_ret.append(returns)
        except:
            pass
        
    bb_strategy_ret_df = pd.DataFrame(bb_strategy_ret).rename(columns = {0:'bb_returns'})

    number_of_stocks = investment_value/solbb['Close'][-1]
    bb_investment_ret = []

    for i in range(len(bb_strategy_ret_df['bb_returns'])):
        returns = number_of_stocks*bb_strategy_ret_df['bb_returns'][i]
        bb_investment_ret.append(returns)

    bb_investment_ret_df = pd.DataFrame(bb_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_investment_ret_df['investment_returns']), 2)
    profit_percentage = math.floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the BB strategy by investing $100k in sol : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the BB strategy : {}%'.format(profit_percentage), attrs = ['bold']))
    return(f'{total_investment_ret}')