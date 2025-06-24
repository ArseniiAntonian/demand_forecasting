import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from KPP_metrik_trend import KPP
import KPP_tan
# === Загрузка и предобработка данных ===
df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/ML.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'])

# Создание лагов и скользящей средней по oil price
df['Oil_Lag1'] = df['Oil_Price'].shift(1)
df['Oil_SMA_3'] = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
# Логарифм целевой переменной
df['Log_Freight'] = np.log1p(df['Freight_Price'])
# Удаляем пропуски
df = df.dropna().reset_index(drop=True)

# Разделение на train/test
df_train = df[df['Date'] < '2020-01-03'].copy()
df_test  = df[(df['Date'] >= '2020-01-03') & (df['Date'] <= '2025-01-03')].copy()

# Подготовка для Prophet
def prepare_df(df_part):
    df_model = df_part[['Date', 'Log_Freight', 'Oil_Lag1', 'Oil_SMA_3', 'Oil_Lag6', 'Oil_Lag12']].copy()
    df_model = df_model.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})
    return df_model

prophet_train = prepare_df(df_train)
prophet_future = prepare_df(df_test)

# Holidays / critical events
# crisis_events = pd.DataFrame({
#     'holiday': ['covid_lockdown','suez_blockage','ukraine_conflict','china_port_closure','inflation_peak'],
#     'ds': pd.to_datetime(['2020-03-15','2021-03-23','2022-02-24','2021-08-01','2022-06-01']),
#     'lower_window': 0,
#     'upper_window': 180
# })
crisis_events = pd.DataFrame({
    'holiday': [
        # из вашего исходного списка
        'covid_lockdown',
        'suez_blockage',
        'ukraine_conflict',
        'china_port_closure',
        'inflation_peak',
        # дополнительные кризисы из датасета
        'covid_19_oil_crash',
        'dot_com_crash',
        'global_financial_crisis',
        'oil_price_collapse',
        'arab_spring',
        'crimea_crisis',
        'iraq_war',
        'iran_sanctions',
        'russia_sanctions',
        'pandemic_covid_19'
    ],
    'ds': pd.to_datetime([
        # ваш список
        '2020-03-15',
        '2021-03-23',
        '2022-02-24',
        '2021-08-01',
        '2022-06-01',
        # даты из датасета
        '2020-03-01',   # COVID-19 oil crash
        '2000-03-10',   # Dot-com crash
        '2008-09-01',   # Global financial crisis
        '2014-10-01',   # Oil price collapse
        '2011-01-01',   # Arab Spring
        '2014-03-01',   # Crimea crisis
        '2003-03-01',   # Iraq War
        '2012-01-01',   # Iran sanctions
        '2014-03-01',   # Russia sanctions
        '2020-03-01'    # COVID-19 pandemic
    ]),
    'lower_window': 0,
    'upper_window': 30
})

# === Обучение Prophet ===
model = Prophet(yearly_seasonality=True, changepoint_prior_scale=2, holidays=crisis_events)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
for feat in ['Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12']:
    model.add_regressor(feat)
model.fit(prophet_train)

# Прогноз для формирования признаков
all_future = pd.concat([prophet_train, prophet_future], axis=0).reset_index(drop=True)
forecast_all = model.predict(all_future)

# Разделяем прогнозы обратно на train/test части
forecast_train = forecast_all.iloc[:len(prophet_train)].copy()
forecast_test  = forecast_all.iloc[len(prophet_train):].copy()

# === Подготовка датасетов для LightGBM ===
merge_feats = ['ds','Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12']
df_lgbm_train = prophet_train[['ds','y'] + merge_feats[1:]].merge(
    forecast_train[['ds','yhat','trend']], on='ds'
)
df_lgbm_train['y_true'] = df_lgbm_train['y']

df_lgbm_test = prophet_future[['ds','y'] + merge_feats[1:]].merge(
    forecast_test[['ds','yhat','trend']], on='ds'
)
df_lgbm_test['y_true'] = df_lgbm_test['y']

# Фичи для LGBM
features = ['yhat','trend','Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12']
X_train = df_lgbm_train[features]
y_train = df_lgbm_train['y_true']
X_test  = df_lgbm_test[features]
y_test  = df_lgbm_test['y_true']

# TimeSeriesSplit для валидации
tscv = TimeSeriesSplit(n_splits=3)

# Обучение LGBM с early stopping через callbacks
lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.03, random_state=42)
for train_idx, valid_idx in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)]
    )

# Предсказание на тестовом наборе
y_pred = lgbm.predict(X_test)

# Обратная экспонента к логарифму
y_pred_exp = np.expm1(y_pred)
y_test_exp = np.expm1(y_test)


print('KPP метрика = ', KPP(y_test_exp,y_pred_exp)*100, ' %')
print('KPP_tan метрика = ', KPP_tan.KPP(y_test_exp,y_pred_exp))
# Оценка метрик
mae = mean_absolute_error(y_test_exp, y_pred_exp)
rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
mape = mean_absolute_percentage_error(y_test_exp, y_pred_exp)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.3f}")

# Визуализация результатов
plt.figure(figsize=(14,6))
plt.plot(df_train['Date'], df_train['Freight_Price'], label='Train',color='blue')
plt.plot(df_test['Date'], df_test['Freight_Price'], label='Test',color='blue')
plt.plot(df_test['Date'], y_pred_exp, '--', label='LGBM+Prophet',color='red')
plt.legend()
plt.title('Объединенный прогноз Prophet + LightGBM')
plt.xlabel('Date')
plt.ylabel('Freight Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
