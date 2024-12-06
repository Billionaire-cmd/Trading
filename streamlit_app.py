import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet  # Advanced time series forecasting tool

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

# Function to create and forecast with Prophet
def prophet_forecasting(data, periods=30):
    # Prepare data for Prophet
    df = data[['Close']].reset_index()
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    
    # Initialize Prophet model
    model = Prophet()
    model.fit(df)
    
    # Create future dataframe and forecast
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast

# Streamlit UI
st.title('🤖🤖🤖💯💯💯 Rabiotic Advanced Technical Analysis & Price Prediction App')

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

# Prophet Forecasting
forecast = prophet_forecasting(market_data, periods=30)

# Prophet Plot
st.write("### Prophet Forecast")
fig_forecast = plt.figure(figsize=(14, 7))
plt.plot(forecast['ds'], forecast['yhat'], label="Forecasted Price", color="orange")
plt.plot(market_data['Close'], label="Historical Close", color="blue")
plt.legend()
plt.title("Forecasted vs Historical Prices")
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(fig_forecast)

# Submit Prediction button
if st.button('Submit Prediction'):
    # Display prediction recommendation
    if predictions[-1] > data['Close'].iloc[-1]:
        st.write("**Recommendation:** Enter a Long Position. Price is expected to rise.")
    else:
        st.write("**Recommendation:** Enter a Short Position. Price is expected to fall.")
