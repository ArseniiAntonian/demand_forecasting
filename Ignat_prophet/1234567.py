import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Загрузка данных
bcti_df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/Baltic Dirty Tanker Historical Data.csv', parse_dates=['Date'])
oil_df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/monthly_oil_cost_1988-2025.csv', parse_dates=['Date'])

# Очистка чисел
bcti_df['Price'] = bcti_df['Price'].astype(str).str.replace(',', '', regex=False).astype(float)
oil_df['Price'] = oil_df['Price'].astype(str).str.replace(',', '', regex=False).astype(float)

# Преобразуем даты к (месяц-год)
bcti_df['Date'] = pd.to_datetime(bcti_df['Date']).dt.to_period('M').dt.to_timestamp()
oil_df['Date'] = pd.to_datetime(oil_df['Date']).dt.to_period('M').dt.to_timestamp()

# Переименование
bcti_df = bcti_df[['Date', 'Price']].rename(columns={'Price': 'Freight_Price'})
oil_df = oil_df[['Date', 'Price']].rename(columns={'Price': 'Oil_Price'})

# Объединение
df = pd.merge(bcti_df, oil_df, on='Date', how='inner').sort_values('Date')

# Признаки
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
df['Freight_Lag6'] = df['Freight_Price'].shift(6)
df['Oil_Roll_3m'] = df['Oil_Price'].rolling(3).mean()
df['Freight_Roll_3m'] = df['Freight_Price'].rolling(3).mean()
df['covid'] = ((df['Date'] >= '2020-03-01') & (df['Date'] <= '2021-06-01')).astype(int)
df['Log_Freight'] = np.log1p(df['Freight_Price'])
df['Oil_Change_1m'] = df['Oil_Price'].pct_change(periods=1)
df['Freight_Change_1m'] = df['Freight_Price'].pct_change(periods=1)
df['Freight_Lag1'] = df['Freight_Price'].shift(1)  # ⬅️ добавлен лаг-1 для baseline

# Убираем пропуски
df = df.dropna().reset_index(drop=True)

# Train/Test split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Prophet: подготовка данных
freight_train = train[['Date', 'Log_Freight', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag6',
                       'Oil_Roll_3m', 'Freight_Roll_3m', 'covid',
                       'Oil_Change_1m', 'Freight_Change_1m']].rename(columns={
    'Date': 'ds',
    'Log_Freight': 'y',
    'Oil_Lag6': 'oil6',
    'Oil_Lag12': 'oil12',
    'Freight_Lag6': 'freight6',
    'Oil_Roll_3m': 'oil_roll',
    'Freight_Roll_3m': 'freight_roll',
    'Oil_Change_1m': 'oil_chg',
    'Freight_Change_1m': 'freight_chg'
})

model = Prophet(
    yearly_seasonality=True,
    changepoint_prior_scale=0.5
)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
model.add_regressor('oil6')
model.add_regressor('oil12')
model.add_regressor('freight6')
model.add_regressor('oil_roll')
model.add_regressor('freight_roll')
model.add_regressor('covid')
model.add_regressor('oil_chg')
model.add_regressor('freight_chg')

# Обучение модели
model.fit(freight_train)

# Подготовка future
future = test[['Date', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag6',
               'Oil_Roll_3m', 'Freight_Roll_3m', 'covid',
               'Oil_Change_1m', 'Freight_Change_1m']].rename(columns={
    'Date': 'ds',
    'Oil_Lag6': 'oil6',
    'Oil_Lag12': 'oil12',
    'Freight_Lag6': 'freight6',
    'Oil_Roll_3m': 'oil_roll',
    'Freight_Roll_3m': 'freight_roll',
    'Oil_Change_1m': 'oil_chg',
    'Freight_Change_1m': 'freight_chg'
})

# Прогноз Prophet
forecast = model.predict(future)
test = test.merge(forecast[['ds', 'yhat']], left_on='Date', right_on='ds', how='left')
test['yhat_exp'] = np.expm1(test['yhat'])

# 📉 Базовый прогноз: просто лаг-1
test['baseline_pred'] = test['Freight_Lag1']

# 📊 Метрики
mae_prophet = mean_absolute_error(test['Freight_Price'], test['yhat_exp'])
rmse_prophet = sqrt(mean_squared_error(test['Freight_Price'], test['yhat_exp']))

mae_baseline = mean_absolute_error(test['Freight_Price'], test['baseline_pred'])
rmse_baseline = sqrt(mean_squared_error(test['Freight_Price'], test['baseline_pred']))

# 📈 Визуализация
plt.figure(figsize=(12, 6))
plt.plot(train['Date'], train['Freight_Price'], label='Train', color='blue')
plt.plot(test['Date'], test['Freight_Price'], label='Test (Факт)', color='green')
plt.plot(test['Date'], test['yhat_exp'], label='Prophet прогноз', color='red', linestyle='--')
plt.plot(test['Date'], test['baseline_pred'], label='Baseline (лаг-1)', color='orange', linestyle=':')
plt.title("Сравнение Prophet vs Базовая модель (лаг 1)")
plt.xlabel("Дата")
plt.ylabel("Фрахтовая ставка")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# 📢 Вывод
print(f"\n📊 MAE Prophet:     {mae_prophet:.2f}")
print(f"📊 RMSE Prophet:    {rmse_prophet:.2f}")
print(f"📉 MAE Baseline:    {mae_baseline:.2f}")
print(f"📉 RMSE Baseline:   {rmse_baseline:.2f}")

improve_mae = 100 * (1 - mae_prophet / mae_baseline)
improve_rmse = 100 * (1 - rmse_prophet / rmse_baseline)
print(f"\n🚀 Prophet улучшил MAE на:  {improve_mae:.1f}%")
print(f"🚀 Prophet улучшил RMSE на: {improve_rmse:.1f}%")
