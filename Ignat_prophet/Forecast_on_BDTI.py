import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# === Загрузка данных ===
bcti_df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/Baltic Dirty Tanker Historical Data.csv', parse_dates=['Date'])
oil_df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/monthly_oil_cost_1988-2025.csv', parse_dates=['Date'])

# === Очистка и подготовка ===
bcti_df['Price'] = bcti_df['Price'].astype(str).str.replace(',', '', regex=False).astype(float)
oil_df['Price'] = oil_df['Price'].astype(str).str.replace(',', '', regex=False).astype(float)

bcti_df['Date'] = pd.to_datetime(bcti_df['Date']).dt.to_period('M').dt.to_timestamp()
oil_df['Date'] = pd.to_datetime(oil_df['Date']).dt.to_period('M').dt.to_timestamp()

bcti_df = bcti_df[['Date', 'Price']].rename(columns={'Price': 'Freight_Price'})
oil_df = oil_df[['Date', 'Price']].rename(columns={'Price': 'Oil_Price'})

df = pd.merge(bcti_df, oil_df, on='Date', how='inner').sort_values('Date')

# === Создание признаков ===
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
df['Freight_Lag6'] = df['Freight_Price'].shift(6)
df['Freight_Lag24'] = df['Freight_Price'].shift(24)

df['Oil_Roll_3m'] = df['Oil_Price'].rolling(3).mean()
df['Freight_Roll_3m'] = df['Freight_Price'].rolling(3).mean()

df['covid'] = ((df['Date'] >= '2020-03-01') & (df['Date'] <= '2021-06-01')).astype(int)

df['Log_Freight'] = np.log1p(df['Freight_Price'])
df['Oil_Change_1m'] = df['Oil_Price'].pct_change(periods=1)
df['Freight_Change_1m'] = df['Freight_Price'].pct_change(periods=1)

df = df.dropna().reset_index(drop=True)

# === Разделение: train — всё до последних 24 мес, test — последние 2 года ===
last_date = df['Date'].max()
cutoff_date = last_date - pd.DateOffset(months=24)

train = df[df['Date'] < cutoff_date].copy()
test = df[df['Date'] >= cutoff_date].copy()

print(f"Train: {train['Date'].min().date()} → {train['Date'].max().date()}")
print(f"Test:  {test['Date'].min().date()} → {test['Date'].max().date()}")

# === Подготовка для Prophet ===
def prepare_df(df_part):
    return df_part[['Date', 'Log_Freight', 'Oil_Lag6', 'Oil_Lag12', 'Oil_Roll_3m',
                    'covid', 'Freight_Lag24']].rename(columns={
        'Date': 'ds',
        'Log_Freight': 'y',
        'Oil_Lag6': 'oil6',
        'Oil_Lag12': 'oil12',
        'Oil_Roll_3m': 'oil_roll',
        #'Oil_Change_1m': 'oil_chg',
        'Freight_Lag24': 'freight24'
    })

freight_train = prepare_df(train)
future = prepare_df(test)

# === Обучение модели ===
model = Prophet(
    yearly_seasonality=True,
    changepoint_prior_scale=0.5
)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
model.add_regressor('oil6')
model.add_regressor('oil12')
model.add_regressor('oil_roll')
model.add_regressor('covid')
model.add_regressor('oil_chg')
model.add_regressor('freight24')
model.fit(freight_train)

# === Прогноз ===
forecast = model.predict(future)
test = test.merge(forecast[['ds', 'yhat']], left_on='Date', right_on='ds', how='left')
test['yhat_exp'] = np.expm1(test['yhat'])

# === Визуализация ===
plt.figure(figsize=(14, 6))
plt.plot(train['Date'], train['Freight_Price'], label='Train', color='blue')
plt.plot(test['Date'], test['Freight_Price'], label='Test (последние 2 года)', color='green')
plt.plot(test['Date'], test['yhat_exp'], '--', label='Forecast', color='red')

plt.title("Прогноз фрахта: Prophet (тест — последние 2 года)")
plt.xlabel("Дата")
plt.ylabel("Фрахтовая ставка")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Метрики ===
mae = mean_absolute_error(test['Freight_Price'], test['yhat_exp'])
rmse = root_mean_squared_error(test['Freight_Price'], test['yhat_exp'])

print("----- Test (последние 2 года) -----")
print(f"MAE : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
