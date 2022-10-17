import os
import pandas as pd
import numpy as np
from numpy import array
import math
import datetime as dt
import matplotlib.pyplot as plt
import json

# For Evalution we will use these library

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM


# For PLotting we will use these library

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

maindf=pd.read_csv('btc.csv')
maindf.shape
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')
names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

#LSTM make
# Lets First Take all the Close Price 
closedf = maindf[['Date','Close']]

#take data 1year
closedf = closedf[closedf['Date'] > '2021-02-19']
close_stock = closedf.copy()


# deleting date column and normalizing using MinMax Scaler
del closedf['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

# keep the training set as 60% and 40% testing se
training_size=int(len(closedf)*0.60)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def model_prediction():
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    #actual model build
    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=32,verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict.shape, test_predict.shape

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

    #Predicting next 30 days
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()


    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 30
    while(i<pred_days):
        
        if(len(temp_input)>time_step):
            
            x_input=np.array(temp_input[1:])
            ##print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            
            yhat = model.predict(x_input, verbose=0)
            ##print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            ##print(temp_input)
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1
                
    #print("Output of predicted next days: ", len(lst_output))
    #plotting last 15 days and next 30 days
    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    #print(last_days)
    #print(day_pred)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })
    return new_pred_plot

def get_rmse():
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    #actual model build
    model=Sequential()
    model.add(LSTM(10,input_shape=(None,1),activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=32,verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    train_predict.shape, test_predict.shape

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

    #Predicting next 30 days
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    rmse = math.sqrt(mean_squared_error(original_ytrain,train_predict))
    return str(rmse)

def get_new_prediction():
    new_pred_plot = model_prediction()
    json_result = new_pred_plot.to_json(orient='index')
    addName = "["+json_result+","+"{\"name\": \"BTC\"}, {\"RMSE\":"+ get_rmse()+"}]"
    return addName