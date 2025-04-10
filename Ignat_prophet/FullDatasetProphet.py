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

# Лаги и признаки
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
df['Freight_Lag6'] = df['Freight_Price'].shift(6)
df['Oil_Roll_3m'] = df['Oil_Price'].rolling(3).mean()
df['Freight_Roll_3m'] = df['Freight_Price'].rolling(3).mean()
df['covid'] = ((df['Date'] >= '2020-03-01') & (df['Date'] <= '2021-06-01')).astype(int)
df['Log_Freight'] = np.log1p(df['Freight_Price'])
df = df.dropna().reset_index(drop=True)

# Ручная временная кросс-валидация по фолдам
start_year = 2000
end_year = 2018
results = []

for train_end in range(2010, end_year + 1, 2):
    test_end = train_end + 2

    train_df = df[(df['Date'] >= f'{start_year}-01-01') & (df['Date'] < f'{train_end}-01-01')].copy()
    test_df = df[(df['Date'] >= f'{train_end}-01-01') & (df['Date'] < f'{test_end}-01-01')].copy()

    freight_train = train_df[['Date', 'Log_Freight', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag6',
                              'Oil_Roll_3m', 'Freight_Roll_3m', 'covid']].rename(columns={
        'Date': 'ds',
        'Log_Freight': 'y',
        'Oil_Lag6': 'oil6',
        'Oil_Lag12': 'oil12',
        'Freight_Lag6': 'freight6',
        'Oil_Roll_3m': 'oil_roll',
        'Freight_Roll_3m': 'freight_roll'
    })

    model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
    model.add_regressor('oil6')
    model.add_regressor('oil12')
    model.add_regressor('freight6')
    model.add_regressor('oil_roll')
    model.add_regressor('freight_roll')
    model.add_regressor('covid')

    model.fit(freight_train)

    future = test_df[['Date', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag6',
                      'Oil_Roll_3m', 'Freight_Roll_3m', 'covid']].rename(columns={
        'Date': 'ds',
        'Oil_Lag6': 'oil6',
        'Oil_Lag12': 'oil12',
        'Freight_Lag6': 'freight6',
        'Oil_Roll_3m': 'oil_roll',
        'Freight_Roll_3m': 'freight_roll'
    })

    forecast = model.predict(future)
    test_df = test_df.merge(forecast[['ds', 'yhat']], left_on='Date', right_on='ds', how='left')
    test_df['yhat_exp'] = np.expm1(test_df['yhat'])

    # Метрики
    mae = mean_absolute_error(test_df['Freight_Price'], test_df['yhat_exp'])
    rmse = sqrt(mean_squared_error(test_df['Freight_Price'], test_df['yhat_exp']))
    results.append({'Train до': train_end, 'Test до': test_end, 'MAE': mae, 'RMSE': rmse})

    # График
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['Date'], train_df['Freight_Price'], label='Train', color='blue')
    plt.plot(test_df['Date'], test_df['Freight_Price'], label='Test (Факт)', color='green')
    plt.plot(test_df['Date'], test_df['yhat_exp'], label='Прогноз', color='red', linestyle='--')
    plt.title(f"Прогноз фрахта: Train до {train_end}, Test {train_end}–{test_end}")
    plt.xlabel("Дата")
    plt.ylabel("Фрахтовая ставка")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Сводка
results_df = pd.DataFrame(results)
print("\nСводка по фолдам:")
print(results_df.to_string(index=False))
