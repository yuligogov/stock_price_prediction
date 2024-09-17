import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# Fetch full day intraday stock data
ticker = "NVDA"
data = yf.download(ticker, start="2024-09-09", end="2024-09-10", interval='1m')
data['Datetime'] = data.index

# Ensure there's enough data
if data.empty:
    raise ValueError("No intraday data found for the specified date range and ticker.")

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Placeholder for practical usage: Fetch real text data (e.g., from news, social media)
data['Sentiment'] = data['Datetime'].apply(lambda x: get_sentiment_score("Placeholder text for practical sentiment data"))

# Technical Indicators
data['SMA'] = data['Close'].rolling(window=20).mean()  # 20-minute SMA
data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()  # 20-minute EMA

# Feature engineering
features = ['Close', 'Volume', 'Sentiment', 'SMA', 'EMA']
data = data.dropna()
X = data[features]
y = data['Close'].shift(-1).dropna()
X = X.iloc[:-1, :]

# Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ensure there is enough data for training and testing
if len(X_scaled) == 0:
    raise ValueError("Not enough data to train the model. Check the date range and data availability.")

# Model training
model = RandomForestRegressor()
model.fit(X_scaled, y)

# Prepare for real-time simulation
data = data.iloc[:-1, :]  # Align data with the shifted y
predictions = model.predict(X_scaled)

# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(14, 7))
actual_line, = ax.plot([], [], label='Actual Prices', color='blue')
predicted_line, = ax.plot([], [], label='Predicted Prices', linestyle='--', color='orange')
ax.set_xlim(data['Datetime'].min(), data['Datetime'].max())
ax.set_ylim(data['Close'].min(), data['Close'].max())
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.title(f'{ticker} Intraday Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

actual_prices = []
predicted_prices = []
timestamps = []

# Trading variables
initial_money = 1000
money = initial_money
stocks = 0

# Function to update the plot and simulate trading
def update_plot(current_index):
    global money, stocks
    timestamp = data['Datetime'].iloc[current_index]
    actual_price = data['Close'].iloc[current_index]
    predicted_price = predictions[current_index]

    actual_prices.append(actual_price)
    predicted_prices.append(predicted_price)
    timestamps.append(timestamp)

    # Trading strategy: buy if predicted price is higher, sell if lower
    if current_index > 0:
        prev_predicted_price = predictions[current_index - 1]
        if predicted_price > prev_predicted_price:
            # Buy as much as possible
            if money > 0:
                stocks = money / actual_price
                money = 0
                print(f"Buying at {timestamp} - Price: {actual_price:.2f}, Stocks: {stocks:.4f}")
        elif predicted_price < prev_predicted_price:
            # Sell all stocks
            if stocks > 0:
                money = stocks * actual_price
                stocks = 0
                print(f"Selling at {timestamp} - Price: {actual_price:.2f}, Money: {money:.2f}")

    actual_line.set_data(timestamps, actual_prices)
    predicted_line.set_data(timestamps, predicted_prices)
    
    ax.set_xlim(min(timestamps), max(timestamps))
    ax.set_ylim(min(min(actual_prices), min(predicted_prices)), max(max(actual_prices), max(predicted_prices)))

    fig.canvas.draw()
    fig.canvas.flush_events()

# Interactive loop
for current_index in range(len(data)):
    update_plot(current_index)
    time.sleep(0.01)  # Adjust this value to control the speed of the simulation

plt.ioff()  # Turn off interactive mode
plt.show()

# Final results
final_money = money + (stocks * actual_prices[-1])
percentage_change = ((final_money - initial_money) / initial_money) * 100
stock_percentage_change = ((actual_prices[-1] - actual_prices[0]) / actual_prices[0]) * 100

print(f"Initial Money: ${initial_money}")
print(f"Final Money: ${final_money:.2f}")
print(f"Percentage Change with Trading: {percentage_change:.2f}%")
print(f"Percentage Change without Trading: {stock_percentage_change:.2f}%")
