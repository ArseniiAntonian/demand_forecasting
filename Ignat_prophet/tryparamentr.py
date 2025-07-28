import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import KPP_metrik_trend as KPP_metrik_trend
import KPP_tan
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Загрузка и предобработка данных ===
df = pd.read_csv(
    '/Users/ignat/Desktop/Demand/demand_forecasting/data/ML_with_crisis.csv',
    parse_dates=['Date']
)

# === Признаки по нефтяным ценам ===
df['Oil_Lag1']      = df['Oil_Price'].shift(1)
df['Oil_Lag2']      = df['Oil_Price'].shift(2)
df['Oil_SMA_3']     = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6']      = df['Oil_Price'].shift(6)
df['Oil_Lag12']     = df['Oil_Price'].shift(12)
df['Oil_Change_1m'] = df['Oil_Price'].pct_change(periods=1)

# === Лог-преобразование целевой переменной ===
df['Log_Freight'] = np.log1p(df['Freight_Price'])

# === Удаляем пропуски по ключевым фичам ===
df = df.dropna(subset=[
    'Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12','Oil_Change_1m','Log_Freight'
]).reset_index(drop=True)

# === Stationary Wavelet Transform (SWT) для Log_Freight ===
wavelet = 'db4'
max_level = pywt.swt_max_level(len(df['Log_Freight']))
level = min(3, max_level)
swt_coeffs = pywt.swt(df['Log_Freight'].values, wavelet, level=level)

detail_feats = []
for i in range(1, level+1):
    df[f'detail{i}'] = swt_coeffs[i-1][1]
    detail_feats.append(f'detail{i}')
approx_feat = f'approx{level}'
df[approx_feat] = swt_coeffs[level-1][0]

# === Разделение на train/test ===
train = df[df['Date'] < '2020-01-03'].copy()
test  = df[(df['Date'] >= '2020-01-03') & (df['Date'] <= '2025-01-03')].copy()

# === Список регрессоров ===
used_features = [
    'Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12','Oil_Change_1m',
    'crisis_intensity','crisis_shock'
] + detail_feats + [approx_feat]

# === Подготовка DataFrame для Prophet ===
def prepare_df(df_part):
    cols = ['Date', 'Log_Freight'] + used_features
    return df_part[cols].rename(columns={'Date':'ds','Log_Freight':'y'})

freight_train = prepare_df(train)
future        = prepare_df(test)

# === Определение кризисных событий ===
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

# === Инициализация и обучение Prophet ===
model = Prophet(
    growth='linear',
    yearly_seasonality=True,
    seasonality_prior_scale=5,
    changepoint_prior_scale=0.1,
    holidays_prior_scale=10,
    holidays=crisis_events,
    mcmc_samples=0           # MCMC отключаем, остаётся только точечная MAP-оценка
)
# Дополнительные сезонности
model.add_seasonality(name='weekly',  period=7,    fourier_order=3)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
# Загрузка регрессоров
for feat in used_features:
    model.add_regressor(feat)

model.fit(freight_train)

# === Опциональная укороченная кросс-валидация (для быстроты) ===
# df_cv = cross_validation(
#     model,
#     initial='365 days',
#     period='90 days',
#     horizon='90 days'
# )
# df_p = performance_metrics(df_cv)
# print(df_p[['horizon','mae','rmse']])

# === Прогноз и оценка на тесте ===
forecast = model.predict(future)
test = test.merge(
    forecast[['ds','yhat','trend']],
    left_on='Date', right_on='ds', how='left'
)
test['yhat_exp'] = np.expm1(test['yhat'])

# Метрики KPP
print('KPP метрика   =',
      KPP_metrik_trend.KPP(test['Freight_Price'], test['yhat_exp']) * 100, '%')
print('KPP_tan метрика =',
      KPP_tan.KPP(test['Freight_Price'], test['yhat_exp']))

# MAE и RMSE
mae  = mean_absolute_error(test['Freight_Price'], test['yhat_exp'])
rmse = np.sqrt(mean_squared_error(test['Freight_Price'], test['yhat_exp']))
print("----- Test (2020-2025) -----")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# === Визуализация результатов ===
plt.figure(figsize=(14,6))
plt.plot(train['Date'], train['Freight_Price'], label='Тренировочные данные')
plt.plot(test['Date'], test['Freight_Price'], label='Тест (2020–2025)')
plt.plot(test['Date'], test['yhat_exp'], '--', label='Прогноз')
plt.title("Прогноз фрахта: Prophet + SWT + регрессоры crisis")
plt.xlabel("Дата")
plt.ylabel("Фрахтовая ставка")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
