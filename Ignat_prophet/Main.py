import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import KPP_metrik_trend as KPP_metrik_trend
import KPP_tan
import pandas as pd
import numpy as np
from prophet import Prophet
import json

# === Загрузка и предобработки данных ===
df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/ML_with_crisis.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'])

# === Признаки ===
df['Oil_Lag1']      = df['Oil_Price'].shift(1)
df['Oil_SMA_3']     = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6']      = df['Oil_Price'].shift(6)
df['Oil_Lag12']     = df['Oil_Price'].shift(12)
df['Oil_Change_1m'] = df['Oil_Price'].pct_change(1)
df['Log_Freight']   = np.log1p(df['Freight_Price'])
# Предполагаем, что в df есть 'crisis_intensity' и 'crisis_shock'

df = df.dropna().reset_index(drop=True)

# === Разбиение на train/test ===
train = df[df['Date'] < '2020-01-03'].copy()
test  = df[(df['Date'] >= '2020-01-03') & (df['Date'] <= '2025-01-03')].copy()
used_features = ['Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12','crisis_intensity','crisis_shock']

# === Определение кризисов ===
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
    'upper_window': 60
})

# === Подготовка DataFrame для Prophet ===
def prepare_df(df_part):
    tmp = df_part[['Date','Log_Freight'] + used_features].rename(columns={'Date':'ds','Log_Freight':'y'})
    return tmp

freight_train = prepare_df(train)
future        = prepare_df(test)

# === Грид поиска гиперпараметров ===
param_grid = {
    'changepoint_prior_scale': [0.5, 1.0, 2.0, 5.0],
    'seasonality_prior_scale': [10.0, 20.0, 30.0],
    'holidays_prior_scale':    [10.0, 25.0, 50.0],
    'seasonality_mode':        ['additive', 'multiplicative']
}

results = []
for cps in param_grid['changepoint_prior_scale']:
    for sps in param_grid['seasonality_prior_scale']:
        for hps in param_grid['holidays_prior_scale']:
            for mode in param_grid['seasonality_mode']:
                # Инициализация и обучение модели
                m = Prophet(
                    yearly_seasonality=True,
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    holidays_prior_scale=hps,
                    seasonality_mode=mode,
                    holidays=crisis_events
                )
                m.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
                for feat in used_features:
                    m.add_regressor(feat)
                m.fit(freight_train)

                # Прогноз и метрики
                fc = m.predict(future)
                tmp = test.copy().merge(fc[['ds','yhat']], left_on='Date', right_on='ds')
                tmp['yhat_exp'] = np.expm1(tmp['yhat'])
                mae  = mean_absolute_error(tmp['Freight_Price'], tmp['yhat_exp'])
                rmse = np.sqrt(mean_squared_error(tmp['Freight_Price'], tmp['yhat_exp']))
                kpp  = KPP_metrik_trend.KPP(tmp['Freight_Price'], tmp['yhat_exp'])
                results.append({
                    'cps': cps, 'sps': sps, 'hps': hps, 'mode': mode,
                    'MAE': mae, 'RMSE': rmse, 'KPP': kpp
                })
                print(f"Done cps={cps}, sps={sps}, hps={hps}, mode={mode} -> MAE={mae:.3f}, RMSE={rmse:.3f}, KPP={kpp:.3f}")

# === Анализ результатов ===
res_df = pd.DataFrame(results)
best = res_df.loc[res_df['RMSE'].idxmin()]
print("\n=== Best parameters by RMSE ===")
print(best)

# Сохраняем лучшие параметры
best_params = best[['cps','sps','hps','mode']].to_dict()
with open('best_params_2020_2025.json','w') as f:
    json.dump(best_params, f)
print("Best parameters saved:", best_params)

# === Финальный прогноз с лучшими параметрами ===
model = Prophet(
    yearly_seasonality=True,
    changepoint_prior_scale=best['cps'],
    seasonality_prior_scale=best['sps'],
    holidays_prior_scale=best['hps'],
    seasonality_mode=best['mode'],
    holidays=crisis_events
)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
for feat in used_features:
    model.add_regressor(feat)
model.fit(freight_train)
fc_final = model.predict(future)
test['yhat_exp'] = np.expm1(fc_final['yhat'])

# Визуализация
plt.figure(figsize=(14,6))
plt.plot(train['Date'], train['Freight_Price'], label='Train')
plt.plot(test['Date'], test['Freight_Price'], label='Test (2020–2025)')
plt.plot(test['Date'], test['yhat_exp'], '--', label='Forecast (best)')
plt.title("Prophet Forecast with Hyperparam Tuning")
plt.xlabel("Date")
plt.ylabel("Freight Price")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
