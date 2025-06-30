import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

from belG.tcn_model import build_model
from belG.tcn_w_exogs import build_model as build_model_exogs

# MODEL_PATH= 'belG/tcn_weights.h5'
# SCALER_X_PATH= 'belG/scaler_X.pkl'
# SCALER_Y_PATH= 'belG/scaler_y.pkl'

MODEL_PATH = 'belG/weights_w_exogs_5y/tcn_weights.h5'
SCALER_X_PATH = 'belG/weights_w_exogs_5y/scaler_X.pkl'
SCALER_Y_PATH = 'belG/weights_w_exogs_5y/scaler_y.pkl'

N_INPUT = 12
N_OUTPUT = 60
FREQ = 'M'

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df['Oil_Lag6']     = df['Oil_Lag2'].shift(4)
    df['Freight_Lag6'] = df['Freight_Lag2'].shift(4)
    df.dropna(inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df

# def forecast_last_data() -> pd.DataFrame:
#     """
#     Загружает модель и скейлеры, делает прогноз на следующие n_output периодов,
#     используя последние n_input точек из df, и возвращает DataFrame с датами и прогнозом.
#     """

#     df = load_data('data/ML.csv')

#     time_cols = [
#         'Oil_Price',
#         'Freight_Lag1','Freight_Lag2',
#         'Oil_Lag1','Oil_Lag2',
#         'Oil_Lag6','Freight_Lag6'
#     ]
#     cat_cols = [
#         'has_crisis','has_war'
#     ]

#     feature_cols = time_cols + cat_cols

#     # Загрузка модели и скейлеров
#     with open('belG/best_params.json', 'r') as f:
#         best_params = json.load(f)

#     scaler_X = joblib.load(SCALER_X_PATH)
#     scaler_y = joblib.load(SCALER_Y_PATH)

#     # Берем последние n_input строк признаков
#     last_X = df[feature_cols].iloc[-N_INPUT:].values
#     print(df[feature_cols].iloc[-N_INPUT:].columns)
#     # Масштабируем
#     last_X_scaled = scaler_X.transform(last_X)
#     # Формируем батч (1, n_input, n_features)
#     seq = last_X_scaled.reshape(1, N_OUTPUT, -1)
#     model = build_model(
#         N_INPUT, last_X_scaled.shape[1], N_OUTPUT,
#         best_params['enc_filters'],
#         best_params['enc_kernel_size'],
#         best_params['enc_dilations'],
#         best_params['enc_dropout'],
#         best_params['dec_filters'],
#         best_params['dec_kernel_size'],
#         best_params['dec_dilations'],
#         best_params['dec_dropout'],
#         best_params['learning_rate'],
#         k_attention=best_params['k_attention']
#     )
#     model.load_weights(MODEL_PATH)
#     # Прогноз в масштабе
#     pred_scaled = model.predict(seq)
#     # Инвертируем скейлинг
#     pred = scaler_y.inverse_transform(pred_scaled[0])

#     # Генерируем даты для прогноза
#     last_date = df.index[-1]
#     forecast_dates = pd.date_range(
#         start=last_date + pd.tseries.frequencies.to_offset(FREQ),
#         periods=N_OUTPUT,
#         freq=FREQ
#     )

#     # Собираем DataFrame с результатами
#     df_forecast = pd.DataFrame({
#         'Forecast': pred.flatten()
#     }, index=forecast_dates)

#     return df_forecast, df

def forecast_last_data_w_exogs(df_exogs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делает прогноз на следующие N_OUTPUT периодов с использованием
    последней обученной модели Seq2Seq с экзогенами.
    Возвращает:
      - df_forecast: DataFrame с прогнозом и датами,
      - df_hist:     исходный исторический DataFrame.
    """
    # 1) История
    df_hist = load_data('data/ML_with_crisis.csv')

    # 2) Оригинальные списки признаков
    time_cols = [
        'Freight_Price', 'Oil_Price',
        'Freight_Lag1','Freight_Lag2',
        'Oil_Lag1','Oil_Lag2',
        'Oil_Lag6','Freight_Lag6',
    ]
    # при обучении мы убирали таргет 'Freight_Price' из X
    feat_cols = [c for c in time_cols if c != 'Freight_Price']
    cat_cols = [
        'has_crisis','crisis_intensity','crisis_shock',
        'crisis_type_Financial','crisis_type_Pandemic',
        'crisis_type_Geopolitical','crisis_type_Natural',
        'crisis_type_Logistical'
    ]

    # 3) Загружаем скейлеры и модель (те же пути, что вы использовали в save_artifacts)
    scaler_X = joblib.load(SCALER_X_PATH)     # путь из вашего модуля: '.../scaler_X.pkl'
    scaler_y = joblib.load(SCALER_Y_PATH)
    with open('belG/best_params_exogs_5y.json', 'r') as f:
        best_params = json.load(f)

    # 4) Берём последние N_INPUT строк по тем же 7 фичам, что и на тренировке
    last_X = df_hist[feat_cols].iloc[-N_INPUT:].astype(np.float32).values
    # теперь scaler_X.n_features_in_ == last_X.shape[1] == 7
    enc_seq = scaler_X.transform(last_X).reshape(1, N_INPUT, len(feat_cols))

    # 5) Даты прогноза
    last_date = df_hist.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(),
        periods=N_OUTPUT, freq='MS'
    )

    # 6) Экзогены
    df_exogs = df_exogs.copy()
    df_exogs['Date'] = pd.to_datetime(df_exogs['Date'])
    df_exogs.set_index('Date', inplace=True)
    future_exog = (
        df_exogs
        .reindex(forecast_dates, method='ffill')[cat_cols]
        .astype(np.float32)
        .values
    )
    exog_seq = future_exog.reshape(1, N_OUTPUT, len(cat_cols))

    # 7) Собираем и грузим TCN-модель
    model = build_model_exogs(
    n_input               = N_INPUT,
    n_features            = len(feat_cols),
    n_output              = N_OUTPUT,
    n_exog                = len(cat_cols),
    enc_filters           = best_params['enc_filters'],
    enc_kernel_size       = best_params['enc_kernel_size'],
    enc_dilations_groups  = best_params['enc_dilations'],
    enc_dropout           = best_params['enc_dropout'],
    dec_filters           = best_params['dec_filters'],
    dec_kernel_size       = best_params['dec_kernel_size'],
    dec_dilations_groups  = best_params['dec_dilations'],
    dec_dropout           = best_params['dec_dropout'],
    learning_rate         = best_params['learning_rate'],
    k_attention           = best_params['k_attention'],
    # если нужно, здесь же можно задать w_financial и прочие
    )
    model.load_weights(MODEL_PATH)

    # 8) Прогноз и развёртка
    pred_scaled = model.predict([enc_seq, exog_seq])[0]     # (N_OUTPUT, 1)
    pred = scaler_y.inverse_transform(pred_scaled)          # (N_OUTPUT, 1)

    # 9) Итоговый DataFrame
    df_forecast = pd.DataFrame(
        {'Forecast': pred.flatten()},
        index=forecast_dates
    )

    return df_forecast, df_hist

