import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Загрузка данных ===
df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/ML.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'])

# === Лаги и скользящие показатели ===
df['Freight_Lag24'] = df['Freight_Price'].shift(24)
df['Oil_Lag1'] = df['Oil_Price'].shift(1)
df['Oil_Lag2'] = df['Oil_Price'].shift(2)
df['Oil_SMA_3'] = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
df['Oil_Change_1m'] = df['Oil_Price'].pct_change(periods=1)

# === Логарифм целевой переменной ===
df['Log_Freight'] = np.log1p(df['Freight_Price'])

# === Удаляем пропуски ===
df = df.dropna().reset_index(drop=True)

# === Единый флаг кризиса ===
crisis_periods = [
    ['2008-09-15', 180],
    ['2018-07-06', 90],
    ['2020-01-23', 120],
    ['2020-09-01', 180]
]

df['is_crisis'] = 0
for start_date, duration in crisis_periods:
    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=duration)
    df['is_crisis'] |= df['Date'].between(start, end).astype(int)

# === Разделение на train/test ===
#train = df[df['Date'] < '2017-01-03'].copy()
train = df[(df['Date'] >= '2008-01-03') & (df['Date'] < '2017-01-03')].copy()
test = df[(df['Date'] >= '2017-01-03') & (df['Date'] <= '2019-01-03')].copy()

# === Используемые признаки ===
used_features = [
    'Oil_Lag1',
    'Oil_SMA_3',
    'Oil_Lag6',
    'Oil_Lag12',
    'Freight_Lag24',
    'is_crisis'
]

# === Подготовка данных для Prophet ===
def prepare_df(df_part):
    df_model = df_part[['Date', 'Log_Freight'] + used_features].copy()
    df_model = df_model.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})
    return df_model

freight_train = prepare_df(train)
future = prepare_df(test)

# === Обучение Prophet ===
model = Prophet(
    yearly_seasonality=True,
    changepoint_prior_scale=0.03,
    seasonality_prior_scale=5.0,
    seasonality_mode='additive',
    n_changepoints=55
)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

for feature in used_features:
    model.add_regressor(feature)

model.fit(freight_train)

# === Прогноз ===
forecast = model.predict(future)
test = test.merge(forecast[['ds', 'yhat']], left_on='Date', right_on='ds', how='left')
test['yhat_exp'] = np.expm1(test['yhat'])

# === Визуализация ===
plt.figure(figsize=(14, 6))
plt.plot(train['Date'], train['Freight_Price'], label='Train', color='blue')
plt.plot(test['Date'], test['Freight_Price'], label='Test (2018–2020)', color='blue')
plt.plot(test['Date'], test['yhat_exp'], '--', label='Forecast', color='red')
plt.title("Прогноз фрахта: Prophet + единый флаг кризиса")
plt.xlabel("Дата")
plt.ylabel("Фрахтовая ставка")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Метрики ===
mae = mean_absolute_error(test['Freight_Price'], test['yhat_exp'])
rmse = np.sqrt(mean_squared_error(test['Freight_Price'], test['yhat_exp']))  # <-- исправлено


print("----- Test (2017–2019) -----")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
