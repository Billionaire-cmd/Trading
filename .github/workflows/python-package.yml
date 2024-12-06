import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Sidebar for selecting trading pair and timeframe
st.sidebar.header("Select Trading Pair and Timeframe")

# List of trading pairs
symbols = [
    'GBPJPY=X',   # GBP/JPY
    'USDJPY=X',   # USD/JPY
    'XAUUSD=X',   # XAU/USD (Gold/USD)
    'US30=X',     # US30 (Dow Jones Index)
    'GBPUSD=X'    # GBP/USD
]

# Sidebar selection
selected_symbol = st.sidebar.selectbox("Choose a trading pair", symbols)

# Timeframe options in minutes
timeframe_options = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "4 Hours": "4h"
}

# Sidebar selection for timeframe
selected_timeframe = st.sidebar.selectbox("Choose a timeframe", list(timeframe_options.keys()))

# Mapping the selected timeframe to the yfinance string format
yf_timeframe = timeframe_options[selected_timeframe]

# Function to fetch market data for the selected trading pair and timeframe
def fetch_market_data(symbol, period="1d", interval="1m"):
    data = yf.download(symbol, period=period, interval=interval)
    return data

# Fetch the data based on the selected symbol and timeframe
market_data = fetch_market_data(selected_symbol, period="1d", interval=yf_timeframe)

# Function to plot the closing prices
def plot_closing_prices(data, symbol, timeframe):
    plt.figure(figsize=(12, 6))
    data['Close'].plot(label=f'{symbol} Close Price')
    plt.title(f'{symbol} Closing Price - {timeframe} Interval')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display the plot in Streamlit

# Plot the closing prices for the selected trading pair and timeframe
plot_closing_prices(market_data, selected_symbol, selected_timeframe)

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to fetch data and perform machine learning-based analysis
def get_data(symbol='AAPL', start_date='2020-01-01'):
    # Fetch historical data using yfinance
    data = yf.download(symbol, start=start_date)
    
    # Calculate technical indicators
    data['RSI'] = calculate_rsi(data['Close'])
    data['SMA'] = data['Close'].rolling(window=50).mean()
    data['EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    # Machine learning prediction model - features and labels
    features = data[['RSI', 'SMA', 'EMA']].dropna()
    target = data['Close'].shift(-1).dropna()  # Predict the next day's close price
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model: Random Forest Regressor (basic ML model)
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train_scaled, y_train)
    
    # Predicting future price
    predictions = rf_model.predict(X_test_scaled)
    
    return data, predictions

# Function to create a deep learning LSTM model for prediction
def create_lstm_model(data):
    # Reshape data for LSTM (time series prediction)
    data = data[['Close']].values
    data = data.reshape((data.shape[0], 1, data.shape[1]))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(data.shape[1], data.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Streamlit UI
st.title('Advanced Technical Analysis & Price Prediction')

# Fetch data and machine learning model predictions
data, predictions = get_data(selected_symbol, "2020-01-01")

# Display data and technical indicators
st.write(f"### {selected_symbol} Data and Technical Indicators")
st.write(data.tail())

# Plotting technical indicators and closing price
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(data['Close'], label='Close Price', color='blue')
ax.plot(data['SMA'], label='50-Day SMA', color='red')
ax.plot(data['EMA'], label='50-Day EMA', color='green')
ax.set_title(f"{selected_symbol} - Price with Indicators")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Plot predictions
fig2, ax2 = plt.subplots(figsize=(14, 7))
ax2.plot(data.index[-len(predictions):], predictions, label='Predicted Price', color='orange')
ax2.plot(data['Close'].tail(len(predictions)), label='Actual Price', color='blue')
ax2.set_title(f"{selected_symbol} - Price Prediction vs Actual")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# LSTM Model Prediction
if st.button('Train LSTM Model'):
    model = create_lstm_model(data)
    data_for_lstm = data[['Close']].values
    data_for_lstm = data_for_lstm.reshape((data_for_lstm.shape[0], 1, data_for_lstm.shape[1]))
    
    model.fit(data_for_lstm, data['Close'], epochs=5, batch_size=32)
    lstm_predictions = model.predict(data_for_lstm)
    
    st.write(f"LSTM Model Prediction for {selected_symbol}:")
    st.write(lstm_predictions[-1])

# Displaying technical analysis insights based on RSI
if data['RSI'].iloc[-1] <= 9:
    st.write("**RSI indicates Strong Buy (LL Entry)**")
    st.write("**Action: Execute Long Position**")
elif data['RSI'].iloc[-1] >= 90:
    st.write("**RSI indicates Strong Sell (HH Entry)**")
    st.write("**Action: Execute Short Position**")
elif data['RSI'].iloc[-1] == 50:
    st.write("**RSI indicates Take Profit (Resistance)**")
    st.write("**Action: Consider Taking Profit**")
elif data['RSI'].iloc[-1] == 80:
    st.write("**RSI indicates Strong Sell (LH Entry)**")
    st.write("**Action: Prepare to Exit Position**")
else:
    st.write("**RSI indicates neutral position or hold**")
    st.write("**Action: Hold Position for Now**")

# Trade Scaling Recommendations based on predictions
if predictions[-1] > data['Close'].iloc[-1]:
    st.write("**Trade Execution Recommendation:** Enter Long Position. Price is expected to rise.")
    st.write("**Scaling Strategy:** Consider increasing position size as price moves upward towards predicted target.")
else:
    st.write("**Trade Execution Recommendation:** Enter Short Position. Price is expected to fall.")
    st.write("**Scaling Strategy:** If price drops as predicted, consider adding more to the short position for higher gains.")

# Exit Strategy based on LSTM and Technical Indicators
if lstm_predictions[-1] > data['Close'].iloc[-1]:
    st.write("**Exit Strategy:** If price continues upward, scale out of position near resistance levels.")
else:
    st.write("**Exit Strategy:** If price continues downward, consider closing short positions near support levels.")
