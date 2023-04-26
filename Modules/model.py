import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt



# Prophet Model
def prophet_model(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    model = Prophet()
    model.fit(train_data)

    future = model.make_future_dataframe(periods=len(test_data), freq='D', include_history=False)
    forecast = model.predict(future)

    y_true = test_data['y'].values
    y_pred = forecast['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    plt.plot(data['ds'], data['y'], label='Actual Test Data')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs. Predicted Stock Prices')
    plt.show()

    model = Prophet()
    model.fit(data)

    prediction_test = model.predict(test_data[['ds']])

    plt.figure(figsize=(10, 6))
    plt.plot(train_data['ds'], train_data['y'], label='Train Actual', color='blue')
    plt.plot(test_data['ds'], test_data['y'], label='Test Actual', color='green')
    plt.plot(test_data['ds'], prediction_test['yhat'], label='Test Predictions', color='orange', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Predictions')
    plt.legend()
    plt.show()




# LSTM Model
def lstm_model(X_train, y_train, epochs=10, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model

# Plot LSTM Results
def plot_lstm_results(test_data, predicted_prices):
    plt.plot(test_data['Date'], test_data['Close'], label='Actual Prices')
    plt.plot(test_data['Date'], predicted_prices, label='Predicted Prices', linestyle='--', color='green')
    plt.title('Stock Price Prediction using LSTM')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


# Auto Regression Model
def fit_auto_reg(df, column, lags=5):
    model = AutoReg(df[column], lags=lags)
    res = model.fit()
    print(res.summary())
    print("μ={} ,ϕ={}".format(res.params[0], res.params[1]))
    return res

def fit_auto_reg_cov_HC0(df, column, lags=5):
    model = AutoReg(df[column], lags=lags)
    res = model.fit(cov_type="HC0")
    print(res.summary())
    print("μ={} ,ϕ={}".format(res.params[0], res.params[1]))
    return res

def plot_auto_reg_diagnostics(res, lags=30, figsize=(16, 9)):
    fig = plt.figure(figsize=figsize)
    res.plot_diagnostics(fig=fig, lags=lags)
    plt.show()
