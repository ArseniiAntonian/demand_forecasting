import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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

# Лаги
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
df['Freight_Lag6'] = df['Freight_Price'].shift(6)

# Скользящие средние (rolling features)
df['Oil_Roll_3m'] = df['Oil_Price'].rolling(3).mean()
df['Freight_Roll_3m'] = df['Freight_Price'].rolling(3).mean()

df['covid'] = ((df['Date'] >= '2020-03-01') & (df['Date'] <= '2021-06-01')).astype(int)

# Логарифм целевой переменной
df['Log_Freight'] = np.log1p(df['Freight_Price'])

# Убираем пропуски
df = df.dropna().reset_index(drop=True)

# Train/Test split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Подготовка train
freight_train = train[['Date', 'Log_Freight', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag6', 'Oil_Roll_3m', 'Freight_Roll_3m', 'covid']].rename(columns={
    'Date': 'ds',
    'Log_Freight': 'y',
    'Oil_Lag6': 'oil6',
    'Oil_Lag12': 'oil12',
    'Freight_Lag6': 'freight6',
    'Oil_Roll_3m': 'oil_roll',
    'Freight_Roll_3m': 'freight_roll'
})

# Prophet модель
model = Prophet(
    yearly_seasonality=True,
    changepoint_prior_scale=0.5
)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

# Добавляем регрессоры
model.add_regressor('oil6')
model.add_regressor('oil12')
model.add_regressor('freight6')
model.add_regressor('oil_roll')
model.add_regressor('freight_roll')
model.add_regressor('covid')

# Обучение
model.fit(freight_train)

# Подготовка future
future = test[['Date', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag6', 'Oil_Roll_3m', 'Freight_Roll_3m', 'covid']].rename(columns={
    'Date': 'ds',
    'Oil_Lag6': 'oil6',
    'Oil_Lag12': 'oil12',
    'Freight_Lag6': 'freight6',
    'Oil_Roll_3m': 'oil_roll',
    'Freight_Roll_3m': 'freight_roll'
})

# Прогноз
forecast = model.predict(future)

# Объединяем с тестом
test = test.merge(forecast[['ds', 'yhat']], left_on='Date', right_on='ds', how='left')
test['yhat_exp'] = np.expm1(test['yhat'])  # возвращаем из логарифма

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(np.array(train['Date']), np.array(train['Freight_Price']), label='Train (Исторические)', color='blue')
plt.plot(np.array(test['Date']), np.array(test['Freight_Price']), label='Test (Фактические)', color='green')
plt.plot(np.array(test['Date']), np.array(test['yhat_exp']), label='Прогноз Prophet (log→exp)', linestyle='dashed', color='red')

plt.xlabel("Дата")
plt.ylabel("Фрахтовая ставка (Freight Price)")
plt.title("Прогноз с лог-преобразованием + скользящими средними + лагами + COVID")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
mae = mean_absolute_error(test['Freight_Price'], test['yhat_exp'])
rmse = root_mean_squared_error(test['Freight_Price'], test['yhat_exp'])


print(f"MAE (без доп. признаками): {mae:.2f}")
print(f"RMSE (без доп. признаками): {rmse:.2f}")