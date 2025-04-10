import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

import matplotlib.pyplot as plt

data = pd.read_csv('data/freight_cost.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date', ascending=True)

p_value = adfuller(data['Price'])[1]
print(p_value)

data['Price_diff'] = data['Price'].diff(12)

p_value_diff = adfuller(data['Price_diff'].dropna())[1]
print(p_value_diff)

data['Price_diff2'] = data['Price_diff'].diff()
p_value_diff2 = adfuller(data['Price_diff2'].dropna())[1]
print(p_value_diff2)


tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(data):
    train, test = data.iloc[train_index], data.iloc[test_index]
    model = SARIMAX(train['Price_diff2'], 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 12), 
                enforce_stationarity=False, 
                enforce_invertibility=False)
    sarima_result = model.fit(disp=False)

    forecast = sarima_result.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    mse = mean_squared_error(test['Price'], forecast_mean)
    mae = mean_absolute_error(test['Price'], forecast_mean)
    mape = mean_absolute_percentage_error(test['Price'], forecast_mean)
    print(mape)
    print(mae)
    print(mse)

plt.figure(figsize=(10, 6))
plt.plot(train['Price'], label='Train')
plt.plot(test['Price'], label='Test', color='orange')
plt.plot(forecast_mean, label='Forecast', color='green')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], 
                 color='green', alpha=0.2)
plt.legend()
plt.title('SARIMA Model Forecast')
plt.show()