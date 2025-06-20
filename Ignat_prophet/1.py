import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def forecast_prophet(data):
    # === Загрузка и базовая подготовка ===
    df_hist = pd.read_csv('data/ML.csv', parse_dates=['Date'])
    df_hist['Date'] = pd.to_datetime(df_hist['Date'])

    if data is not None:
        df = pd.concat([df_hist, data], ignore_index=True)
    else:
        df = df_hist.copy()

    df['Date'] = pd.to_datetime(df['Date'])

    # === Фичи для всех данных ===
    df['Freight_Lag24']   = df['Freight_Price'].shift(24)
    df['Oil_Lag1']        = df['Oil_Price'].shift(1)
    df['Oil_Lag2']        = df['Oil_Price'].shift(2)
    df['Oil_SMA_3']       = df['Oil_Price'].rolling(3).mean()
    df['Oil_Lag6']        = df['Oil_Price'].shift(6)
    df['Oil_Lag12']       = df['Oil_Price'].shift(12)
    df['Oil_Change_1m']   = df['Oil_Price'].pct_change(periods=1)
    df['Log_Freight']     = np.log1p(df['Freight_Price'])

    # === Формируем события по всем 1-кам ===
    war_starts    = df.loc[df['has_war'] == 1, 'Date']
    crisis_starts = df.loc[df['has_crisis'] == 1, 'Date']

    events = pd.concat([
        pd.DataFrame({
            'holiday':      'war',
            'ds':           war_starts,
            'lower_window': 0,
            'upper_window': 180
        }),
        pd.DataFrame({
            'holiday':      'crisis',
            'ds':           crisis_starts,
            'lower_window': 0,
            'upper_window': 180
        })
    ], ignore_index=True)
    print(events.tail(10))

    # === Убираем пропуски перед обучением ===
    df = df.dropna().reset_index(drop=True)

    # === Делим на train и будущие данные ===
    train = df[df['Date'] <= '2025-01-03'].copy()
    used_features = [
        'Oil_Lag1', 'Oil_SMA_3',
        'Oil_Lag6', 'Oil_Lag12',
        'Freight_Lag24'
    ]

    # Обрезаем data до макс. горизонта
    max_horizon = 24
    if data is not None and len(data) > max_horizon:
        data = data.iloc[:max_horizon].copy()

    # Собираем фичи для будущего по тем же правилам
    if data is not None:
        n = len(data)
        full_oil = pd.concat([df['Oil_Price'], data['Oil_Price']], ignore_index=True)
        full_freight = pd.concat([df['Freight_Lag24'], pd.Series([np.nan] * n)], ignore_index=True)

        data['Oil_Lag1']      = full_oil.shift(1).iloc[-n:].values
        data['Oil_Lag2']      = full_oil.shift(2).iloc[-n:].values
        data['Oil_SMA_3']     = full_oil.rolling(3).mean().iloc[-n:].values
        data['Oil_Lag6']      = full_oil.shift(6).iloc[-n:].values
        data['Oil_Lag12']     = full_oil.shift(12).iloc[-n:].values
        data['Oil_Change_1m'] = full_oil.pct_change(1).iloc[-n:].values
        data['Freight_Lag24'] = full_freight.shift(24).iloc[-n:].values
    else:
        data = pd.DataFrame(columns=['Date'] + used_features)

    # Подготовка для Prophet
    def prepare_train(df_part):
        m = df_part[['Date', 'Log_Freight'] + used_features].copy()
        return m.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})

    def prepare_future(df_part):
        m = df_part[['Date'] + used_features].copy()
        return m.rename(columns={'Date': 'ds'})

    freight_train = prepare_train(train)
    future        = prepare_future(data)

    # === Обучение модели ===
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=2,
        holidays=events if not events.empty else None
    )
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
    for feat in used_features:
        model.add_regressor(feat)
    model.fit(freight_train)

    # === Прогноз ===
    forecast = model.predict(future)
    forecast['yhat_exp'] = np.expm1(forecast['yhat'])

    return train, forecast
