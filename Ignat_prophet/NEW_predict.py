import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from prophet import Prophet

def forecast_prophet(data):
    df_hist = pd.read_csv('data/ML_with_crisis.csv', parse_dates=['Date'])
    df_hist['Date'] = pd.to_datetime(df_hist['Date'])
    if data is not None:
        df = pd.concat([df_hist, data], ignore_index=True)
    else:
        df = df_hist.copy()

    df['Date'] = pd.to_datetime(df['Date'])
    # === Фичи ===
    df['Oil_Lag1']      = df['Oil_Price'].shift(1)
    df['Oil_Lag2']      = df['Oil_Price'].shift(2)
    df['Oil_SMA_3']     = df['Oil_Price'].rolling(3).mean()
    df['Oil_Lag6']      = df['Oil_Price'].shift(6)
    df['Oil_Lag12']     = df['Oil_Price'].shift(12)
    df['Oil_Change_1m'] = df['Oil_Price'].pct_change(periods=1)
    df['Log_Freight']   = np.log1p(df['Freight_Price'])

    train = df[df['Date'] <= '2025-01-03'].copy()
    test = df[(df['Date'] > '2025-01-03') & (df['Date'] <= '2030-01-03')].copy()
    
    used_features = [
        'Oil_Lag1',
        'Oil_SMA_3',
        'Oil_Lag6',
        'Oil_Lag12',
        'crisis_intensity',
        'crisis_shock'
    ]

    if data is not None:
        n = len(data)
        full_oil = pd.concat([df['Oil_Price'], data['Oil_Price']], ignore_index=True)

        print(df['Oil_Price'].head())
        data['Oil_Lag1']      = full_oil.shift(1).iloc[-n:].values
        data['Oil_Lag2']      = full_oil.shift(2).iloc[-n:].values
        data['Oil_SMA_3']     = full_oil.rolling(3).mean().iloc[-n:].values
        data['Oil_Lag6']      = full_oil.shift(6).iloc[-n:].values
        data['Oil_Lag12']     = full_oil.shift(12).iloc[-n:].values
        data['Oil_Change_1m'] = full_oil.pct_change(1).iloc[-n:].values

        data['Log_Freight'] = np.nan
        test = data.copy()

    

    
   
    def prepare_df(df_part):
        df_model = df_part[['Date', 'Log_Freight'] + used_features].copy()
        df_model = df_model.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})
        return df_model

    freight_train = prepare_df(train).dropna()
    future = prepare_df(test)

    # === DEBUG (оставь пока нужно) ===
    print("freight_train shape:", freight_train.shape)
    if freight_train.shape[0] < 2:
        raise ValueError("Недостаточно данных для обучения модели Prophet. Проверь входные данные и признаки.")

    # === Праздники ===
    crisis_events = pd.DataFrame({
        'holiday': [
            'covid_lockdown','suez_blockage','ukraine_conflict','china_port_closure','inflation_peak',
            'covid_19_oil_crash','dot_com_crash','global_financial_crisis','oil_price_collapse',
            'arab_spring','crimea_crisis','iraq_war','iran_sanctions','russia_sanctions','pandemic_covid_19'
        ],
        'ds': pd.to_datetime([
            '2020-03-15','2021-03-23','2022-02-24','2021-08-01','2022-06-01',
            '2020-03-01','2000-03-10','2008-09-01','2014-10-01',
            '2011-01-01','2014-03-01','2003-03-01','2012-01-01','2014-03-01','2020-03-01'
        ]),
        'lower_window': 0,
        'upper_window': 30
    })

    model = Prophet(
        yearly_seasonality=True,
        holidays_prior_scale=25,
        changepoint_prior_scale=2,
        holidays=crisis_events
    )
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

    for feature in used_features:
        model.add_regressor(feature)

    model.fit(freight_train)

    forecast = model.predict(future)
    forecast['yhat_exp'] = np.expm1(forecast['yhat'])

    result = test.copy()
    result = result.merge(forecast[['ds', 'yhat', 'trend', 'yhat_exp']], left_on='Date', right_on='ds', how='left')

    return train, result
