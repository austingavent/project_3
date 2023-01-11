# Initial imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

  # Importing required Keras modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def window_data(trading_signals_df, window):
    """
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(trading_signals_df) - window):
        features = trading_signals_df.iloc[i : (i + window), :]
        target = trading_signals_df.iloc[(i + window), -1]
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y).reshape(-1, 1)


def main(symbol_df, window_size):

    #Create new trading signals Df, Set index as datetime object and drop extraneous columns
    trading_signals_df = pd.DataFrame()

    #add daily change rates to increase the staionarity of dataset
    trading_signals_df['volume delta'] = symbol_df['Volume'].dropna().pct_change()
    trading_signals_df['bb std delta'] = symbol_df['bollinger_std'].dropna().pct_change()
    trading_signals_df['rvol delta'] = symbol_df['rvol'].dropna().pct_change()
    trading_signals_df['option rvol delta'] = symbol_df['Option rVol'].dropna().pct_change()
    trading_signals_df['Option Signal'] = symbol_df['Option Signal'].dropna().pct_change()
    #add daily returns as target
    trading_signals_df['daily returns'] = symbol_df['daily returns'].dropna()

    trading_signals_df= trading_signals_df.fillna(value = 0)
    trading_signals_df= trading_signals_df.replace([np.inf, -np.inf], 0.0)
    
    X, y = window_data(trading_signals_df, window_size)
    
    split = int(0.7 * len(X))

    X_train = X[: split]
    X_test = X[split:]

    y_train = y[: split]
    y_test = y[split:]
    
    scalers = {}
    for i in range(X_train.shape[1]):
        scalers[i] = MinMaxScaler()
        X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :]) 

    for i in range(X_test.shape[1]):
        X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])
        
                    
    scaler = MinMaxScaler()
    scaler.fit(y)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    # Define the Transformer model.
    model = Sequential()

    # Initial model setup
    number_units = 7
    dropout_fraction = 0.3
    inputs = len(trading_signals_df.columns)

    # Layer 1
    model.add(LSTM(
        units=number_units,
        return_sequences=True,
        input_shape=(X.shape[1], inputs))
        )
    model.add(Dropout(dropout_fraction))

    # Layer 2
    model.add(LSTM(units=number_units, return_sequences=True))
    model.add(Dropout(dropout_fraction))

    # Layer 3
    model.add(LSTM(units=number_units))
    model.add(Dropout(dropout_fraction))

    # Output layer
    model.add(Dense(1, activation = 'linear'))
    
    print(model.summary())
    
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, y_train, epochs=100, shuffle=False, verbose=0)
    
    predicted = model.predict(X_test)
    
    predicted_prices = scaler.inverse_transform(predicted)
    
    real_prices = scaler.inverse_transform(y_test.reshape(-1,1))
    
    shift =7
    
    stocks = pd.DataFrame({
        "Actual": real_prices.ravel(),
        "Predicted": predicted_prices.ravel()
    }, index = trading_signals_df.index[-len(real_prices)-shift:-shift ] )


    # Show the DataFrame's head
    #convert stocks df into positive and negative signals
    stocks['Positive Actual signal'] = np.where(stocks['Actual'] > 0, 1, 0)
    stocks['Negative Actual signal'] = np.where(stocks['Actual'] < 0, -1, 0)

    stocks['Positive Predicted signal'] = np.where(stocks['Predicted'] > 0, 1, 0)
    stocks['Negative Predicted signal'] = np.where(stocks['Predicted'] < 0, -1, 0)

    #merge to create one column per signal, shifted back to reflect forward projection window
    shift = 7
    stocks['Actual Signal'] = stocks['Positive Actual signal'] + stocks['Negative Actual signal']
    stocks['LSTM Predicted Signal'] = stocks['Positive Predicted signal'] + stocks['Negative Predicted signal']
    
    return stocks