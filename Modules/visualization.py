import numpy as np 
import pandas as pd 
import os
from matplotlib.pyplot import Figure
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Distribution Plots
def plot_distribution(df, column, xlabel='Date', ylabel='Prices', figsize=(18, 4)):
    plt.figure(figsize=figsize)
    df[column].plot(label=column)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# Moving Average Plots
def plot_moving_average(df, column, window=50, xlabel='Date', ylabel='Prices', figsize=(18, 5)):
    df['Moving_Average'] = df[column].rolling(window).mean()
    plt.figure(figsize=figsize)
    df['Moving_Average'].plot(label='Moving Average')
    df[column].plot(label=column)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Multiple Line Plots
def plot_multiline_chart(df, columns, xlabel='Date', ylabel='Prices', figsize=(18, 5)):
    plt.subplots(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    for col in columns:
        df[col].plot(label=col)
    
    plt.legend()
    plt.title('Stock Prices')
    plt.show()

# Price Return Plot
def plot_price_return(df, column, figsize=(18, 5)):
    df['PriceDifference'] = df[column].shift(-1) - df[column]
    df['ReturnPrice'] = df['PriceDifference'] / df[column]
    df['ReturnPrice'].plot(figsize=figsize)
    plt.title('Price Return', size=15)
    plt.show()


# Severeal Moving Average Plots
def plot_several_moving_averages(df, column, windows, xlabel='Date', ylabel='Prices', figsize=(18, 5)):
    for window in windows:
        column_name = f'MovingAverage for {window} days'
        df[column_name] = df[column].rolling(window).mean()
    
    columns_to_plot = [column] + [f'MovingAverage for {window} days' for window in windows]
    df[columns_to_plot].plot(subplots=False, figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Rate of Change Plot
def plot_rate_of_change(df, column, windows, xlabel='Date', ylabel='Rate of Change', figsize=(18, 9)):
    change_columns = []
    for window in windows:
        column_name = f'Change{window}'
        df[column_name] = df[column].pct_change(window)
        change_columns.append(column_name)

    for i in range(0, len(change_columns), 2):
        df[change_columns[i:i + 2]].plot(figsize=figsize, legend=True, linestyle='--', marker='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


# Seasonal Decomposition Plot
def plot_seasonal_decomposition(df, column, model='additive', period=1):
    result = seasonal_decompose(df[column], model=model, period=period)
    result.plot()
    plt.show()


# Lag Plot
def plot_lag(df, column):
    lag_plot(df[column])
    plt.show()


# Autocorrelation Plot
def plot_autocorrelation(df, column, column_diff, lags=24, figsize=(12, 8)):
    # Non-stationary original data
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(df[column].dropna(), lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(df[column].dropna(), lags=lags, ax=ax2)
    plt.show()

    # Stationary data
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(df[column_diff].dropna(), lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(df[column_diff].dropna(), lags=lags, ax=ax2)
    plt.show()