import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load your time-series dataset (assuming data has a DateTime index and 'solar_power_output' column)
data = pd.read_csv("C:\\Users\\150LAB\\Desktop\\03f4d1c1a55947025601.csv")

# Ensure the data is sorted by timestamp (important for time-series analysis)
data = data.sort_index()

# Plot the solar power output data
plt.figure(figsize=(10,6))
plt.plot(data[data.columns[0]])
plt.title('Solar Power Output Over Time')
plt.xlabel('Time')
plt.ylabel('Solar Power Output')
plt.show()

# Step 2: Differencing (if necessary to make series stationary)
# We can check if differencing is required by checking the stationarity of the series
# For simplicity, we'll assume the data is stationary or pre-differenced

# Step 3: Fit ARIMA Model
# Select the ARIMA order (p, d, q) based on the dataset
# (p=AR terms, d=differencing order, q=MA terms)
model = ARIMA(data['solar_power_output'], order=(5,1,0))  # Example order (p=5, d=1, q=0)
model_fit = model.fit()

# Step 4: Make predictions
forecast = model_fit.forecast(steps=24)  # Forecast for the next 24 hours
print(forecast)

# Step 5: Evaluate Model (optional)
# You can compute the RMSE on a test set if available
# For demonstration, we just compute RMSE on a known part of the series
predictions = model_fit.predict(start='2025-02-01', end='2025-02-02')
rmse = sqrt(mean_squared_error(data[data.columns[0]]['2025-02-01':'2025-02-02'], predictions))
print(f'RMSE: {rmse}')

# Step 6: Visualize the forecast
plt.figure(figsize=(10,6))
plt.plot(data[data.columns[0]], label='Historical Data')
plt.plot(pd.date_range(data.index[-1], periods=25, freq='H')[1:], forecast, label='Forecast', color='red')
plt.title('Solar Power Output Forecast (ARIMA)')
plt.xlabel('Time')
plt.ylabel('Solar Power Output')
plt.legend()
plt.show()
