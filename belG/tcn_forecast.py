import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

from belG.tcn_model import build_model

MODEL_PATH= 'belG/tcn_weights.h5'
SCALER_X_PATH= 'belG/scaler_X.pkl'
SCALER_Y_PATH= 'belG/scaler_y.pkl'
N_INPUT = 24
N_OUTPUT = 24
FREQ = 'M'

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df['Oil_Lag6']     = df['Oil_Lag2'].shift(4)
    df['Freight_Lag6'] = df['Freight_Lag2'].shift(4)
    df.dropna(inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df

def forecast_last_data() -> pd.DataFrame:
    """
    Загружает модель и скейлеры, делает прогноз на следующие n_output периодов,
    используя последние n_input точек из df, и возвращает DataFrame с датами и прогнозом.
    """

    df = load_data('data/ML.csv')

    time_cols = [
        'Oil_Price',
        'Freight_Lag1','Freight_Lag2',
        'Oil_Lag1','Oil_Lag2',
        'Oil_Lag6','Freight_Lag6'
    ]
    cat_cols = [
        'has_crisis','has_war'
    ]

    feature_cols = time_cols + cat_cols

    # Загрузка модели и скейлеров
    with open('belG/best_params.json', 'r') as f:
        best_params = json.load(f)

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Берем последние n_input строк признаков
    last_X = df[feature_cols].iloc[-N_INPUT:].values
    print(df[feature_cols].iloc[-N_INPUT:].columns)
    # Масштабируем
    last_X_scaled = scaler_X.transform(last_X)
    # Формируем батч (1, n_input, n_features)
    seq = last_X_scaled.reshape(1, N_OUTPUT, -1)
    model = build_model(
        N_INPUT, last_X_scaled.shape[1], N_OUTPUT,
        best_params['enc_filters'],
        best_params['enc_kernel_size'],
        best_params['enc_dilations'],
        best_params['enc_dropout'],
        best_params['dec_filters'],
        best_params['dec_kernel_size'],
        best_params['dec_dilations'],
        best_params['dec_dropout'],
        best_params['learning_rate'],
        k_attention=best_params['k_attention']
    )
    model.load_weights(MODEL_PATH)
    # Прогноз в масштабе
    pred_scaled = model.predict(seq)
    # Инвертируем скейлинг
    pred = scaler_y.inverse_transform(pred_scaled[0])

    # Генерируем даты для прогноза
    last_date = df.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(FREQ),
        periods=N_OUTPUT,
        freq=FREQ
    )

    # Собираем DataFrame с результатами
    df_forecast = pd.DataFrame({
        'Forecast': pred.flatten()
    }, index=forecast_dates)

    return df_forecast, df
