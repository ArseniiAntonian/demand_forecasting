import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from KPP_metrik_trend import KPP
import KPP_tan
import matplotlib.pyplot as plt

# === 1. Загрузка и предобработка данных ===
path = '/Users/ignat/Desktop/Demand/demand_forecasting/data/ML.csv'
df = pd.read_csv(path, parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'])

# Создаем лаги и скользящие средние по цене нефти
df['Oil_Lag1']  = df['Oil_Price'].shift(1)
df['Oil_SMA_3'] = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6']  = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)

# Лог-преобразование таргета
df['Log_Freight'] = np.log1p(df['Freight_Price'])

# === 2. Определение кризисных событий для Prophet и бинарных фич ===
crisis_events = pd.DataFrame({
    'holiday': [
        'covid_lockdown','suez_blockage','ukraine_conflict','china_port_closure',
        'inflation_peak','covid_19_oil_crash','dot_com_crash','global_financial_crisis',
        'oil_price_collapse','arab_spring','crimea_crisis','iraq_war',
        'iran_sanctions','russia_sanctions','pandemic_covid_19'
    ],
    'ds': pd.to_datetime([
        '2020-03-15','2021-03-23','2022-02-24','2021-08-01','2022-06-01',
        '2020-03-01','2000-03-10','2008-09-01','2014-10-01','2011-01-01',
        '2014-03-01','2003-03-01','2012-01-01','2014-03-01','2020-03-01'
    ]),
    'lower_window': 0,
    'upper_window': 30
})
# Приоритеты событий
weight_map = {
    'dot_com_crash': 1.0,
    'global_financial_crisis': 1.0,
    'inflation_peak': 1.0,
    'covid_lockdown': 5.0,
    'covid_19_oil_crash': 5.0,
    'pandemic_covid_19': 5.0,
    'ukraine_conflict': 8.0,
    'crimea_crisis': 8.0,
    'russia_sanctions': 8.0,
    'iran_sanctions': 8.0,
    'iraq_war': 8.0,
    'arab_spring': 8.0,
    'suez_blockage': 5.0,
    'china_port_closure': 5.0,
    'oil_price_collapse': 1.0
}
crisis_events['prior_scale'] = crisis_events['holiday'].map(weight_map)

# Категории кризисов для бинарных регрессоров
groups = {
    'financial':    ['dot_com_crash','global_financial_crisis','inflation_peak'],
    'pandemic':     ['covid_lockdown','covid_19_oil_crash','pandemic_covid_19'],
    'geopolitical': ['ukraine_conflict','crimea_crisis','russia_sanctions','iran_sanctions','iraq_war','arab_spring'],
    'logistical':   ['suez_blockage','china_port_closure'],
    'natural':      ['oil_price_collapse']
}
# Формируем бинарные признаки
df = df.dropna().reset_index(drop=True)
for grp, evs in groups.items():
    dates = crisis_events[crisis_events['holiday'].isin(evs)]['ds']
    df[f'is_{grp}'] = df['Date'].isin(dates).astype(int)

# === 3. Разделение на train и test по дате ===
split_date = '2020-01-03'
df_train = df[df['Date'] < split_date].copy()
df_test  = df[df['Date'] >= split_date].copy()

# Утилита для подготовки данных под Prophet
def prepare(df_part):
    cols = ['Date','Log_Freight','Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]
    return df_part[cols].rename(columns={'Date':'ds','Log_Freight':'y'})

prophet_train = prepare(df_train)
# prophet_test не нужен отдельно, прогнозируем сразу на df_test в итерации

# === 4. Обучение Prophet ===
best_prophet_params = {'n_changepoints': 100, 'changepoint_prior_scale': 2.0}
prophet_model = Prophet(
    yearly_seasonality=True,
    n_changepoints=best_prophet_params['n_changepoints'],
    changepoint_prior_scale=best_prophet_params['changepoint_prior_scale'],
    holidays=crisis_events
)
prophet_model.add_seasonality('monthly', 30.5, fourier_order=5)
prophet_model.add_seasonality('quarterly', 91.25, fourier_order=3)
for feat in ['Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]:
    prophet_model.add_regressor(feat)
prophet_model.fit(prophet_train)

# === 5. Прогноз на train для фич LGBM ===
forecast_train = prophet_model.predict(prophet_train)
df_lgbm_train = (
    prophet_train[['ds','y'] + ['Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]]
    .merge(forecast_train[['ds','yhat','trend']], on='ds')
)
X_train = df_lgbm_train[['yhat','trend','Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]]
y_train = df_lgbm_train['y']

# === 6. Обучение LGBM на train ===
best_lgbm_params = {'learning_rate': 0.1, 'n_estimators': 500, 'num_leaves': 31}
lgbm_model = LGBMRegressor(**best_lgbm_params, random_state=42)
lgbm_model.fit(X_train, y_train)

# === 7. Итеративный прогноз на test с фичами Prophet ===
# Подготовка будущего фрейма с регрессорами
future = df_test[['Date','Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]].copy()
future = future.rename(columns={'Date':'ds'})

# Получаем прогноз Prophet на все столбцы, включая регрессоры
prophet_fc = prophet_model.predict(future)
future = future.merge(prophet_fc[['ds','yhat','trend']], on='ds')

# Итеративный прогноз LGBM
preds_log = []
for idx, row in future.iterrows():
    X_row = pd.DataFrame([{
        'yhat':      row['yhat'],
        'trend':     row['trend'],
        'Oil_Lag1':  row['Oil_Lag1'],
        'Oil_SMA_3': row['Oil_SMA_3'],
        'Oil_Lag6':  row['Oil_Lag6'],
        'Oil_Lag12': row['Oil_Lag12'],
        **{f'is_{g}': row[f'is_{g}'] for g in groups}
    }])
    pred_log = lgbm_model.predict(X_row)[0]
    preds_log.append(pred_log)

future['y_pred_log'] = preds_log
future['y_pred']     = np.expm1(future['y_pred_log'])

# === 8. Оценка и визуализация ===
y_true = df_test['Freight_Price'].values
y_pred = future['y_pred'].values
print('KPP =',    KPP(y_true, y_pred) * 100, '%')
print('KPP_tan =',KPP_tan.KPP(y_true, y_pred))
print('MAE =',    mean_absolute_error(y_true, y_pred))
print('RMSE =',   np.sqrt(mean_squared_error(y_true, y_pred)))
print('MAPE =',   mean_absolute_percentage_error(y_true, y_pred))

plt.figure(figsize=(14,6))
plt.plot(df_train['Date'], df_train['Freight_Price'], label='Train')
plt.plot(df_test['Date'],  df_test['Freight_Price'],  label='Test')
plt.plot(df_test['Date'],  future['y_pred'],         '--', label='Iterative Forecast')
plt.title('Итеративный прогноз LightGBM с учётом прогноза Prophet')
plt.xlabel('Date')
plt.ylabel('Freight Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

