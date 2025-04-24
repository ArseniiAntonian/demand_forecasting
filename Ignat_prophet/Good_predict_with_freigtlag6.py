import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

if __name__ == "__main__":
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

    # Скользящие средние
    df['Oil_Roll_3m'] = df['Oil_Price'].rolling(3).mean()
    df['Freight_Roll_3m'] = df['Freight_Price'].rolling(3).mean()

    # Флаг COVID
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
    freight_train = train[['Date', 'Log_Freight', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag6',
                           'Oil_Roll_3m', 'Freight_Roll_3m', 'covid']].rename(columns={
        'Date': 'ds',
        'Log_Freight': 'y',
        'Oil_Lag6': 'oil6',
        'Oil_Lag12': 'oil12',
        'Freight_Lag6': 'freight6',
        'Oil_Roll_3m': 'oil_roll',
        'Freight_Roll_3m': 'freight_roll'
    })

    # Prophet модель
    model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.5)
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

    # ========== КРОСС-ВАЛИДАЦИЯ ==========
    horizons = ['730 days', '1095 days', '1460 days']  # 2, 3, 4 года
    for h in horizons:
        print(f"\n=== Cross-validation for horizon: {h} ===")
        df_cv = cross_validation(
            model,
            initial='2190 days',
            period='365 days',
            horizon=h
        )
        df_metrics = performance_metrics(df_cv)
        print(df_metrics[['horizon', 'mape', 'rmse']].groupby('horizon').mean())

        plot_cross_validation_metric(df_cv, metric='mape')
        plt.title(f'MAPE для горизонта {h}')
        plt.tight_layout()
        plt.show()

    # ========== ПРОГНОЗ ==========
    future = test[['Date', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag6',
                   'Oil_Roll_3m', 'Freight_Roll_3m', 'covid']].rename(columns={
        'Date': 'ds',
        'Oil_Lag6': 'oil6',
        'Oil_Lag12': 'oil12',
        'Freight_Lag6': 'freight6',
        'Oil_Roll_3m': 'oil_roll',
        'Freight_Roll_3m': 'freight_roll'
    })

    forecast = model.predict(future)

    # Объединяем с test
    test = test.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], left_on='Date', right_on='ds', how='left')

    # Обратно из логарифма
    test['yhat_exp'] = np.expm1(test['yhat'])
    test['yhat_lower_exp'] = np.expm1(test['yhat_lower'])
    test['yhat_upper_exp'] = np.expm1(test['yhat_upper'])

    # Визуализация с интервалом
    plt.figure(figsize=(14, 6))
    plt.plot(train['Date'], train['Freight_Price'], label='Train (История)', color='blue')
    plt.plot(test['Date'], test['Freight_Price'], label='Test (Факт)', color='green')
    plt.plot(test['Date'], test['yhat_exp'], label='Прогноз (Prophet)', color='red', linestyle='--')
    plt.fill_between(test['Date'], test['yhat_lower_exp'], test['yhat_upper_exp'],
                     color='red', alpha=0.2, label='Доверительный интервал')
    plt.title("Прогноз фрахта с Prophet и интервалами")
    plt.xlabel("Дата")
    plt.ylabel("Фрахтовая ставка")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
