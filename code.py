import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error

# User inputs
company_name = input("Enter the ticker symbol or name of the company: ")
start_date = input("Enter the start date (YYYY-MM-DD) for historical data: ")
end_date = input("Enter the end date (YYYY-MM-DD) for historical data: ")

# Download historical data
data = yf.download(company_name, start=start_date, end=end_date)

# Handle missing values
data.ffill(inplace=True)

# Normalize the 'Close' price data
scaler = MinMaxScaler()
data['Close_scaled'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create new features: Moving Averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Drop NaN values and create returns feature
data.dropna(inplace=True)
data['Returns'] = data['Close'].pct_change()

# Add additional features (example: Volume scaled)
data['Volume_scaled'] = scaler.fit_transform(data['Volume'].values.reshape(-1, 1))

# Select relevant features
features = data[['Close_scaled', 'MA20', 'MA50', 'Returns', 'Volume_scaled']].dropna()

# Linear Regression Model
X = features[['MA20', 'MA50', 'Returns', 'Volume_scaled']]
y = features['Close_scaled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)
predictions_lr = scaler.inverse_transform(predictions_lr.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.values.reshape(-1, 1))

mae_lr = mean_absolute_error(y_test_actual, predictions_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_actual, predictions_lr))

print(f'Linear Regression - MAE: {mae_lr}, RMSE: {rmse_lr}')

# LSTM Model
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X_lstm, y_lstm = create_dataset(features['Close_scaled'].values.reshape(-1, 1), time_step)

X_train_lstm, X_test_lstm = X_lstm[:int(0.8*len(X_lstm))], X_lstm[int(0.8*len(X_lstm)):]
y_train_lstm, y_test_lstm = y_lstm[:int(0.8*len(y_lstm))], y_lstm[int(0.8*len(y_lstm)):]

X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

model_lstm = Sequential()
model_lstm.add(Input(shape=(time_step, 1)))
model_lstm.add(LSTM(50, return_sequences=True))
model_lstm.add(LSTM(50, return_sequences=False))
model_lstm.add(Dense(25))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train_lstm, y_train_lstm, batch_size=1, epochs=1)

predictions_lstm = model_lstm.predict(X_test_lstm)
predictions_lstm = scaler.inverse_transform(predictions_lstm)

y_test_lstm_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))
mae_lstm = mean_absolute_error(y_test_lstm_actual, predictions_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm_actual, predictions_lstm))

print(f'LSTM - MAE: {mae_lstm}, RMSE: {rmse_lstm}')

# Calculate additional key performance metrics
percentage_change = (y_test_actual[-1] - y_test_actual[0]) / y_test_actual[0] * 100
std_deviation = np.std(y_test_actual)
rolling_avg_20 = data['MA20'][-1]
rolling_avg_50 = data['MA50'][-1]

# Fetch additional financial data
ticker = yf.Ticker(company_name)
financials = ticker.info

# Dashboard summary
print("\nDashboard Summary:")
print("------------------")
print(f"Company: {company_name}")
print(f"Date Range: {start_date} to {end_date}")
print(f"Max Actual Price: {max_actual_price}")
print(f"Min Actual Price: {min_actual_price}")
print(f"Avg Actual Price: {avg_actual_price}")
print(f"Max Predicted Price (Linear Regression): {max_predicted_lr}")
print(f"Min Predicted Price (Linear Regression): {min_predicted_lr}")
print(f"Avg Predicted Price (Linear Regression): {avg_predicted_lr}")
print(f"Max Predicted Price (LSTM): {max_predicted_lstm}")
print(f"Min Predicted Price (LSTM): {min_predicted_lstm}")
print(f"Avg Predicted Price (LSTM): {avg_predicted_lstm}")
print(f"Percentage Change: {percentage_change}%")
print(f"Volatility (Standard Deviation): {std_deviation}")
print(f"Rolling Average (20 days): {rolling_avg_20}")
print(f"Rolling Average (50 days): {rolling_avg_50}")
print(f"Previous Close: {financials['previousClose']}")
print(f"Average Volume: {financials['averageVolume']}")
print(f"Market Cap: {financials['marketCap']}")
print(f"Shares Outstanding: {financials['sharesOutstanding']}")
print(f"EPS (TTM): {financials['trailingEps']}")
print(f"P/E (TTM): {financials['trailingPE']}")
print(f"Fwd Dividend (% Yield): {financials['dividendYield']}")
print(f"Ex-Dividend Date: {financials['exDividendDate']}")

# Plotting actual vs predicted prices with max/min points highlighted
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, label='Actual Prices', color='blue', linestyle='-')
plt.plot(predictions_lr, label='Predicted Prices (Linear Regression)', color='green', linestyle='--')
plt.plot(predictions_lstm, label='Predicted Prices (LSTM)', color='orange', linestyle='-.')
plt.scatter(np.argmax(y_test_actual), max_actual_price, color='red', marker='o', label='Max Actual Price')
plt.scatter(np.argmin(y_test_actual), min_actual_price, color='red', marker='o', label='Min Actual Price')
plt.scatter(np.argmax(predictions_lr), max_predicted_lr, color='purple', marker='*', label='Max Predicted LR')
plt.scatter(np.argmin(predictions_lr), min_predicted_lr, color='purple', marker='*', label='Min Predicted LR')
plt.scatter(np.argmax(predictions_lstm), max_predicted_lstm, color='brown', marker='^', label='Max Predicted LSTM')
plt.scatter(np.argmin(predictions_lstm), min_predicted_lstm, color='brown', marker='^', label='Min Predicted LSTM')
plt.title('Actual vs. Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
