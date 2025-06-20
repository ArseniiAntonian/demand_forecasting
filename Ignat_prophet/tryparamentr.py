import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from itertools import product
import random
import warnings

warnings.filterwarnings("ignore")

# === Загрузка и подготовка ===
df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/ML.csv', parse_dates=['Date'])
df['Freight_Lag24'] = df['Freight_Price'].shift(24)
df['Oil_Lag1'] = df['Oil_Price'].shift(1)
df['Oil_SMA_3'] = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
df['Log_Freight'] = np.log1p(df['Freight_Price'])
df = df.dropna().reset_index(drop=True)

train = df[df['Date'] < '2018-01-03'].copy()
test = df[(df['Date'] >= '2018-01-03') & (df['Date'] <= '2020-01-03')].copy()

used_features = ['Oil_Lag1', 'Oil_SMA_3', 'Oil_Lag6', 'Oil_Lag12', 'Freight_Lag24']

def prepare_df(df_part):
    df_model = df_part[['Date', 'Log_Freight'] + used_features].copy()
    df_model = df_model.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})
    return df_model

freight_train = prepare_df(train)
future = prepare_df(test)

# === Полный список кризисов ===
full_crisis_events = pd.DataFrame({
    'holiday': [
        'global_financial_crisis', 'oil_price_crash_2014', 'arab_spring',
        'euro_debt_crisis', 'fukushima_disaster', 'china_export_boom',
        'brexit_vote', 'russia_sanctions', 'us_china_trade_war', 'bab_el_mandeb_threat',
        'china_covid_lockdown', 'global_covid_spread', 'brexit_exit', 'container_crisis',
        'covid_lockdown', 'suez_blockage', 'ukraine_conflict', 'china_port_closure', 'inflation_peak'
    ],
    'ds': pd.to_datetime([
        '2008-09-15', '2014-06-01', '2011-01-01', '2010-05-01', '2011-03-11', '2010-01-01',
        '2016-06-23', '2014-03-01', '2018-07-06', '2018-07-25',
        '2020-01-23', '2020-03-01', '2020-01-31', '2020-09-01',
        '2020-03-15', '2021-03-23', '2022-02-24', '2021-08-01', '2022-06-01'
    ]),
    'lower_window': [0] * 19,
    'upper_window': [
        180, 120, 90, 120, 60, 180, 90, 180, 120, 60,
        90, 120, 90, 180, 180, 90, 180, 90, 165
    ]
})

# Обрезаем до 2020
crisis_pre2020 = full_crisis_events[full_crisis_events['ds'] < '2020-01-01']

# === Сетка гиперпараметров ===
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.03, 0.1, 0.3],
    'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0],
    'holidays_prior_scale': [0.1, 1.0, 5.0],
    'seasonality_mode': ['additive', 'multiplicative']
}
all_params = list(product(*param_grid.values()))
random.shuffle(all_params)

results = []

# === Поиск лучших параметров ===
for crisis_set, crisis_name in zip(
    [full_crisis_events, crisis_pre2020, None],
    ['full', 'pre2020', 'no_holidays']
):
    for cps, sps, hps, sm in all_params[:50]:  # максимум 50 запусков
        try:
            model = Prophet(
                yearly_seasonality=True,
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
                holidays_prior_scale=hps,
                seasonality_mode=sm,
                holidays=crisis_set,
                n_changepoints=50
            )
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
            for f in used_features:
                model.add_regressor(f)
            model.fit(freight_train)
            forecast = model.predict(future)
            y_true = test['Freight_Price'].values
            y_pred = np.expm1(forecast['yhat'].values)
            mae = mean_absolute_error(y_true, y_pred)
            results.append({
                'mae': mae,
                'crisis': crisis_name,
                'changepoint_prior_scale': cps,
                'seasonality_prior_scale': sps,
                'holidays_prior_scale': hps,
                'seasonality_mode': sm
            })
        except Exception as e:
            print(f"Ошибка: {e}")

# === Результаты ===
results_df = pd.DataFrame(results).sort_values('mae')
print("\n=== ТОП-5 по MAE ===")
print(results_df.head())
results_df.to_csv("prophet_hyperparam_results.csv", index=False)
