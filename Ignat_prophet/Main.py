import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import KPP_metrik_trend as KPP_metrik_trend
import KPP_tan
import pandas as pd
import numpy as np
from prophet import Prophet

# === Загрузка и предобработка данных (заменили файл на тот, что содержит Crisis_intensity) ===
df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/ML_with_crisis.csv', parse_dates=['Date'])

# === Преобразование даты и создание лагов/скользящей средней ===
df['Date'] = pd.to_datetime(df['Date'])
df['Oil_Lag1']   = df['Oil_Price'].shift(1)
df['Oil_Lag2']   = df['Oil_Price'].shift(2)
df['Oil_SMA_3']  = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6']   = df['Oil_Price'].shift(6)
df['Oil_Lag12']  = df['Oil_Price'].shift(12)
df['Oil_Change_1m'] = df['Oil_Price'].pct_change(periods=1)
# (предполагается, что в ML_with_crisis.csv уже есть колонка 'Crisis_intensity')

# === Логарифм целевой переменной ===
df['Log_Freight'] = np.log1p(df['Freight_Price'])

# === Удаляем строки с пропусками ===
df = df.dropna().reset_index(drop=True)

# === Разделение на train/test ===
train = df[df['Date'] < '2020-01-03'].copy()
test  = df[(df['Date'] >= '2020-01-03') & (df['Date'] <= '2025-01-03')].copy()

# === Список признаков, включая Crisis_intensity ===
used_features = [
    'Oil_Lag1',
    'Oil_SMA_3',
    'Oil_Lag6',
    'Oil_Lag12',
    'crisis_intensity',
    'crisis_shock'   # <-- добавили новый регрессор
]

# === Подготовка датафрейма для Prophet ===
def prepare_df(df_part):
    df_model = df_part[['Date', 'Log_Freight'] + used_features].copy()
    df_model = df_model.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})
    return df_model

freight_train = prepare_df(train)
future        = prepare_df(test)

# === Определение кризисных событий (как и раньше) ===
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

# === Обучение модели Prophet с новым регрессором ===
model = Prophet(
    yearly_seasonality=True,
    holidays_prior_scale=25,
    changepoint_prior_scale=2,
    holidays=crisis_events
)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

# Добавляем все регрессоры из списка, включая Crisis_intensity
for feature in used_features:
    model.add_regressor(feature)

model.fit(freight_train)

# === Прогноз и оценка ===
forecast = model.predict(future)
test = test.merge(forecast[['ds','yhat','trend']], left_on='Date', right_on='ds', how='left')
test['yhat_exp'] = np.expm1(test['yhat'])

print('KPP метрика   =', KPP_metrik_trend.KPP(test['Freight_Price'], test['yhat_exp'])*100, '%')
print('KPP_tan метрика =', KPP_tan.KPP(test['Freight_Price'], test['yhat_exp']))

# === Визуализация ===
plt.figure(figsize=(14, 6))
plt.plot(train['Date'], train['Freight_Price'], label='Тренировочные данные')
plt.plot(test['Date'], test['Freight_Price'], label='Тест (2020–2025)')
plt.plot(test['Date'], test['yhat_exp'], '--', label='Прогноз')
plt.title("Прогноз фрахта: Prophet + внешний регрессор Crisis_intensity")
plt.xlabel("Дата")
plt.ylabel("Фрахтовая ставка")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Метрики MAE и RMSE
mae  = mean_absolute_error(test['Freight_Price'], test['yhat_exp'])
rmse = np.sqrt(mean_squared_error(test['Freight_Price'], test['yhat_exp']))
print("----- Test (2020-2025) -----")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
