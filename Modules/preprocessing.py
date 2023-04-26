import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler

# Augmented Dickey-Fuller Test
def adfuller_test(series):
    result = adfuller(series)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis, indicating it is non-stationary ")


# Convert data to Stationary
def convert_to_stationary(df, column):
    df[f'{column} First Difference'] = df[column] - df[column].shift(1)
    df.dropna(inplace = True)
    return df.head()


# Preparing data for Prophet
def prepare_prophet_data(df, column):
    data = df[[column]].reset_index().rename(columns={"Date": "ds", column: "y"})
    return data


def prepare_lstm_data(train_data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(window_size, len(train_data_scaled)):
        X_train.append(train_data_scaled[i-window_size:i, 0])
        y_train.append(train_data_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler