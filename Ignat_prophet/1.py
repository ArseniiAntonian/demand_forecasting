import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def forecast_prophet(data):
    # === Загрузка и базовая подготовка ===
    df = pd.read_csv(
        'data/ML.csv',
        parse_dates=['Date']
    )
    df['Date'] = pd.to_datetime(df['Date'])
    
    # === Фичи для исторических данных ===
    df['Freight_Lag24'] = df['Freight_Price'].shift(24)
    df['Oil_Lag1']      = df['Oil_Price'].shift(1)
    df['Oil_Lag2']      = df['Oil_Price'].shift(2)
    df['Oil_SMA_3']     = df['Oil_Price'].rolling(3).mean()
    df['Oil_Lag6']      = df['Oil_Price'].shift(6)
    df['Oil_Lag12']     = df['Oil_Price'].shift(12)
    df['Oil_Change_1m'] = df['Oil_Price'].pct_change(periods=1)
    df['Log_Freight']   = np.log1p(df['Freight_Price'])
    
    # Убираем пропуски в историческом df перед обучением
    df = df.dropna().reset_index(drop=True)

    # === Подготовка train и регрессоров ===
    train = df[df['Date'] <= '2025-01-03'].copy()
    used_features = [
        'Oil_Lag1', 'Oil_SMA_3',
        'Oil_Lag6', 'Oil_Lag12',
        'Freight_Lag24'
    ]

    # === Ограничение горизонта прогноза до 24 периодов ===
    max_horizon = 24
    if len(data) > max_horizon:
        data = data.iloc[:max_horizon].copy()

    # === Векторная подготовка фичей для будущего (data) ===
    n = len(data)
    full_oil = pd.concat([df['Oil_Price'], data['Oil_Price']], ignore_index=True)
    full_freight = pd.concat(
        [df['Freight_Lag24'], pd.Series([np.nan] * n)],
        ignore_index=True
    )
    data['Oil_Lag1']      = full_oil.shift(1).iloc[-n:].values
    data['Oil_Lag2']      = full_oil.shift(2).iloc[-n:].values
    data['Oil_SMA_3']     = full_oil.rolling(3).mean().iloc[-n:].values
    data['Oil_Lag6']      = full_oil.shift(6).iloc[-n:].values
    data['Oil_Lag12']     = full_oil.shift(12).iloc[-n:].values
    data['Oil_Change_1m'] = full_oil.pct_change(1).iloc[-n:].values
    data['Freight_Lag24'] = full_freight.shift(24).iloc[-n:].values
    
    # === Подготовка датафреймов для Prophet ===
    def prepare_train(df_part):
        df_m = df_part[['Date', 'Log_Freight'] + used_features].copy()
        return df_m.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})

    def prepare_future(df_part):
        df_m = df_part[['Date'] + used_features].copy()
        return df_m.rename(columns={'Date': 'ds'})

    freight_train = prepare_train(train)
    future        = prepare_future(data)

    # === События-кризисы ===
    crisis_events = pd.DataFrame({
        'holiday': [
            'covid_lockdown', 'suez_blockage', 'ukraine_conflict',
            'china_port_closure', 'inflation_peak'
        ],
        'ds': pd.to_datetime([
            '2020-03-15', '2021-03-23', '2022-02-24',
            '2021-08-01', '2022-06-01'
        ]),
        'lower_window': 0,
        'upper_window': 180
    })

    # === Обучение модели ===
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=2,
        holidays=crisis_events
    )
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
    for feat in used_features:
        model.add_regressor(feat)
    model.fit(freight_train)

    # === Прогноз ===
    forecast = model.predict(future)
    forecast['yhat_exp'] = np.expm1(forecast['yhat'])   
    # === Формирование test на исторических данных ===
    test = df[
        (df['Date'] >= '2025-01-03') & (df['Date'] <= '2027-03-01')
    ].copy()
    test = test.merge(
        forecast[['ds', 'yhat']],
        left_on='Date', right_on='ds', how='left'
    )
    test['yhat_exp'] = np.expm1(test['yhat'])
    return train, forecast


# === Пример вызова ===
# future_data = pd.DataFrame({'Date': [...], 'Oil_Price': [...], 'Freight_Price': [...]})
# train_df, test_df = forecast_prophet(None, future_data)



# === Пример вызова ===
# future_data = pd.DataFrame({'Date': [...], 'Oil_Price': [...], 'Freight_Price': [...]})
# train_df, test_df = forecast_prophet(None, future_data)
