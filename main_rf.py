import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import requests
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
#import hvplot.pandas
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

#Import SKLearn Library and Classes
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def read_symbol_file(symbol_file):
    symbol_df = symbol_file
    return symbol_df

def process_rf_symbol_file(symbol_df, shift):

    #Create new trading signals Df, Set index as datetime object and drop extraneous columns
    trading_signals_df = pd.DataFrame()

    #create features only dataframe by shifting feature signal, select features as x_var_list (for selling OTM puts, bollinger = bollinger short)
    x_var_list = ['bollinger_short', 'rvol signal', 'Option rVol signal']

    trading_signals_df[x_var_list] = symbol_df[x_var_list].shift(1)

    trading_signals_df = trading_signals_df.set_index(pd.to_datetime(symbol_df.index, infer_datetime_format=True))
    #trading_signals_df['b']

    trading_signals_df['Positive Return'] = np.where(symbol_df['daily returns'].shift(-shift)> 0, 1, 0)
    
    #drop Na and reduce df by shift amount
    trading_signals_df = trading_signals_df.dropna()
    trading_signals_df = trading_signals_df[:-shift]
    
    return trading_signals_df

def rf_split_reshape(symbol_signals_df, split_ratio):

    split = int(split_ratio * len(symbol_signals_df))

    X_train = symbol_signals_df.iloc[: split, :-1]
    X_test = symbol_signals_df.iloc[split:, :-1]

    y_train = symbol_signals_df['Positive Return'][:split]
    y_test = symbol_signals_df['Positive Return'][split:]
    
    return X_train, X_test, y_train, y_test

def run_rm(X_train, X_test, y_train, y_test, symbol_df, shift):
    
    # Fit a SKLearn linear regression using just the training set (X_train, Y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    model.fit(X_train, y_train)

    # Make a prediction of "y" values from the X_test dataset
    predicted_df = model.predict(X_test)
    
    # Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:
    result_df = y_test.to_frame()
    result_df["RF Predicted Value"] = predicted_df
    result_df['Forward Daily Returns'] = symbol_df['daily returns'].shift(-shift)
    
    print()
    
    return result_df

def main_rf(symbol_file):

    symbol_df = read_symbol_file(symbol_file)

    shift = 7
    symbol_signals_df = process_rf_symbol_file(symbol_df, shift)

    split_ratio = 0.7
    X_train, X_test, y_train, y_test = rf_split_reshape(symbol_signals_df, split_ratio)

    result_rf = run_rm(X_train, X_test, y_train, y_test, symbol_df, shift)
    return result_rf

#symbol_file = "etf_MTUM_initial.csv"

#result = main_rf(symbol_file)
#result.head(30)









#Results['RF Predicted Value'].to_pickle(r'C:\Users\Kiel\Desktop\FINTECH\UCB_fintech_homework\project_2 - local\Resources\RF_signals_df.pickle')

#Results[['Positive Return', 'Predicted Value']].plot(figsize=(20,10), kind = 'bar')

#Results['Positive Return'].sum()

# Calculate cumulative return of model and plot the result
#(1 + (Results['Forward Daily Returns'] * Results['Predicted Value'])).cumprod().plot()

# Set initial capital allocation
#initial_capital = 100000

# Plot cumulative return of model in terms of capital
#cumulative_return_capital = initial_capital * (1 + (Results['Forward Daily Returns'] * Results['Predicted Value'])).cumprod()
#cumulative_return_capital.plot()

