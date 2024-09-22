# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Simulate Sales Data (Daily Sales for 2 Years)
np.random.seed(42)
date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

# Simulating a time series with a trend and seasonality (TikTok Shop sales)
sales = 1000 + 10 * np.arange(len(date_range)) + 500 * np.sin(np.linspace(0, 3 * np.pi, len(date_range))) + np.random.normal(0, 100, len(date_range))

# Create a DataFrame to hold sales data
sales_data = pd.DataFrame({'date': date_range, 'sales': sales})

# Plot the simulated sales data
plt.figure(figsize=(12, 6))
plt.plot(sales_data['date'], sales_data['sales'], label='Sales')
plt.title('Simulated TikTok Shop Sales Data (Daily)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 2: Prepare Data for Time-Series Analysis
# Set the date as the index for time-series analysis
sales_data.set_index('date', inplace=True)

# Split data into training and test sets
train_data = sales_data.iloc[:-60]  # Use all but the last 60 days for training
test_data = sales_data.iloc[-60:]  # Test on the last 60 days

# Step 3: Fit the ARIMA Model
# We'll start by using an ARIMA(5, 1, 0) model (based on typical p, d, q parameters)
model = ARIMA(train_data['sales'], order=(5, 1, 0))  # ARIMA(p, d, q)
arima_result = model.fit(disp=False)

# Step 4: Forecast Future Sales
# Forecast for the next 60 days
forecast, stderr, conf_int = arima_result.forecast(steps=60)

# Convert the forecast into a DataFrame for easier plotting
forecast_index = test_data.index
forecast_df = pd.DataFrame({
    'date': forecast_index,
    'forecast': forecast,
    'lower_conf': conf_int[:, 0],
    'upper_conf': conf_int[:, 1]
}).set_index('date')

# Step 5: Evaluate the Model
mae = mean_absolute_error(test_data['sales'], forecast_df['forecast'])
rmse = np.sqrt(mean_squared_error(test_data['sales'], forecast_df['forecast']))
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Step 6: Visualize the Actual vs Forecasted Sales
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['sales'], label='Train Data')
plt.plot(test_data.index, test_data['sales'], label='Test Data', color='orange')
plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='green')
plt.fill_between(forecast_df.index, forecast_df['lower_conf'], forecast_df['upper_conf'], color='gray', alpha=0.3)
plt.title('TikTok Shop Sales Forecast vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
