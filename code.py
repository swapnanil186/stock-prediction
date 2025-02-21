import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error

# User inputs
company_name = input("Enter the ticker symbol or name of the company: ")
start_date = input("Enter the start date (YYYY-MM-DD) for historical data: ")
end_date = input("Enter the end date (YYYY-MM-DD) for historical data: ")

# Download historical data
data = yf.download(company_name, start=start_date, end=end_date)
data.ffill(inplace=True)  # Fill missing values

# Normalize 'Close' price
data['Close_scaled'] = MinMaxScaler().fit_transform(data['Close'].values.reshape(-1, 1))

# Feature engineering
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['Returns'] = data['Close'].pct_change()
data['Volume_scaled'] = MinMaxScaler().fit_transform(data['Volume'].values.reshape(-1, 1))
data.dropna(inplace=True)

# Define features and target
features = data[['MA20', 'MA50', 'Returns', 'Volume_scaled']]
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, shuffle=False)

# Baseline Model: Naive Forecast (Previous day's price)
baseline_pred = y_test.shift(1).fillna(method='bfill')
mae_baseline = mean_absolute_error(y_test, baseline_pred)
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))

# Linear Regression Model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, predictions_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, predictions_lr))

# LSTM Model (Sequential)
def create_lstm_dataset(data, time_step=100):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

time_step = 100
X_lstm, y_lstm = create_lstm_dataset(data['Close_scaled'].values.reshape(-1, 1), time_step)
X_train_lstm, X_test_lstm = X_lstm[:int(0.8 * len(X_lstm))], X_lstm[int(0.8 * len(X_lstm)):]
y_train_lstm, y_test_lstm = y_lstm[:int(0.8 * len(y_lstm))], y_lstm[int(0.8 * len(y_lstm)) :]

X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

model_lstm = Sequential([
    Input(shape=(time_step, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train_lstm, y_train_lstm, batch_size=1, epochs=10, verbose=0)

predictions_lstm = model_lstm.predict(X_test_lstm)
predictions_lstm = MinMaxScaler().fit(data[['Close']]).inverse_transform(predictions_lstm)

mae_lstm = mean_absolute_error(y_test_lstm, predictions_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, predictions_lstm))

# Calculate % improvement over baseline
improvement_lr = ((mae_baseline - mae_lr) / mae_baseline) * 100
improvement_lstm = ((mae_baseline - mae_lstm) / mae_baseline) * 100

print("\nModel Performance:")
print("----------------------------------")
print(f"Baseline - MAE: {mae_baseline:.2f}, RMSE: {rmse_baseline:.2f}")
print(f"Linear Regression - MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}, Improvement: {improvement_lr:.2f}%")
print(f"LSTM - MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}, Improvement: {improvement_lstm:.2f}%")

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(predictions_lr, label='Predicted (Linear Regression)', color='green', linestyle='--')
plt.plot(predictions_lstm, label='Predicted (LSTM)', color='orange', linestyle='-.')
plt.title(f'Stock Price Prediction for {company_name}')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
