import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def forecast_lgb(df_pred: pd.DataFrame):
    # 1) Загрузка и предобработка истории
    df_train = pd.read_csv('data/ML_with_crisis.csv')

    df_train = df_train[['Date', 'Freight_Price', 'Oil_Price', 'sin_month', 'cos_month',
       'crisis_intensity', 'crisis_shock', 'crisis_type_Financial',
       'crisis_type_Pandemic', 'crisis_type_Geopolitical',
       'crisis_type_Natural', 'crisis_type_Logistical']]
    
    df_train["Date"] = pd.to_datetime(df_train["Date"], format="%Y-%m-%d")
    df_train.sort_values("Date", inplace=True)
    df_train.set_index("Date", inplace=True)
    df_train.index.freq = 'MS'
    
    for lag in [1, 3, 12]:
        df_train[f'Freight_Price(t-{lag})'] = df_train['Freight_Price'].shift(lag)
        df_train[f'Oil_Price(t-{lag})'] = df_train['Oil_Price'].shift(lag)
    df_train = df_train.dropna()

    X = df_train.drop(columns=['Freight_Price'])
    y = df_train['Freight_Price']

    # 3) Train/Test и обучение модели (ваши гиперпараметры)
    X_train, X_test = X[:-60], X[-60:]
    y_train, y_test = y[:-60], y[-60:]
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # y_pred_test = model.predict(X_test)
    # print(f"MAE={mean_absolute_error(y_test, y_pred_test):.3f}, "
    #       f"MSE={mean_squared_error(y_test, y_pred_test):.3f}, "
    #       f"R2={r2_score(y_test, y_pred_test):.3f}")

    # 4) Подготовка df_pred
    df_pred = df_pred.loc[:, ~df_pred.columns.str.startswith('Unnamed')]
    df_pred['Date'] = pd.to_datetime(df_pred['Date'], format="%Y-%m-%d")
    df_pred.set_index('Date', inplace=True)
    df_pred.index.freq = 'MS'
    df_pred = df_pred.loc[:'2030-05-01']
    forecast_dates = df_pred.index

    # df_pred.drop(columns='Unnamed: 0', axis=1, inplace=True)
    # 5) Объединяем историю и экзогены
    df_full = pd.concat([df_train, df_pred], axis=0).sort_index()
    # df_full['Month'] = df_full.index.month
    df_full['sin_month'] = np.sin(2 * np.pi * df_full.index.month / 12)
    df_full['cos_month'] = np.cos(2 * np.pi * df_full.index.month / 12)

    # 5.1) Генерируем нефть-лаги только для прогнозного периода
    for lag in [1, 3, 12]:
        df_full.loc[forecast_dates, f'Oil_Price(t-{lag})'] = df_full['Oil_Price'].shift(lag)

    # 6) Итеративный прогноз по датам из df_pred
    freight_lags = [1, 3, 12]
    feature_cols = X.columns.tolist()  # набор признаков, как при обучении

    for date in forecast_dates:
        # динамически заполняем лаги Freight_Price из истории и предыдущих предсказаний
        for lag in freight_lags:
            prev = date - pd.DateOffset(months=lag)
            df_full.at[date, f'Freight_Price(t-{lag})'] = df_full.at[prev, 'Freight_Price']

        feat = df_full.loc[date, feature_cols]
        if feat.isna().any():
            miss = feat[feat.isna()].index.tolist()
            raise ValueError(f"Пропуски в признаках на {date.date()}: {miss}")

        df_full.at[date, 'Freight_Price'] = model.predict(feat.values.reshape(1, -1))[0]

    # 7) Отдаём прогнозы
    y_forecast = df_full.loc[forecast_dates, 'Freight_Price']
    return y, y_forecast
