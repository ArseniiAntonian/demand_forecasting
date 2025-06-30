import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from KPP_metrik_trend import KPP
import KPP_tan
import matplotlib.pyplot as plt

# === 1. Загрузка и предобработка ===
path = '/Users/ignat/Desktop/Demand/demand_forecasting/data/ML.csv'
df = pd.read_csv(path, parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'])
# Лаги и скользящие средние
df['Oil_Lag1'] = df['Oil_Price'].shift(1)
df['Oil_SMA_3'] = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
# Логарифм целевого
df['Log_Freight'] = np.log1p(df['Freight_Price'])

# === 2. Кризисные события и веса ===
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
# Весовая карта кризисов
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

# Группы для бинарных фич
groups = {
    'financial': ['dot_com_crash','global_financial_crisis','inflation_peak'],
    'pandemic': ['covid_lockdown','covid_19_oil_crash','pandemic_covid_19'],
    'geopolitical': ['ukraine_conflict','crimea_crisis','russia_sanctions','iran_sanctions','iraq_war','arab_spring'],
    'logistical': ['suez_blockage','china_port_closure'],
    'natural': ['oil_price_collapse']
}
# Добавляем бинарные признаки
for grp, events in groups.items():
    dates = crisis_events[crisis_events['holiday'].isin(events)]['ds']
    df[f'is_{grp}'] = df['Date'].isin(dates).astype(int)

# Удаляем пропуски после лагов
df = df.dropna().reset_index(drop=True)

# === 3. Разделение на train/test ===
split_date = '2020-01-03'
df_train = df[df['Date'] < split_date].copy()
df_test  = df[df['Date'] >= split_date].copy()

# Функция подготовки для Prophet
def prepare(df_part):
    cols = ['Date','Log_Freight','Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]
    return df_part[cols].rename(columns={'Date':'ds','Log_Freight':'y'})
prophet_train = prepare(df_train)
prophet_test  = prepare(df_test)

# === 4. Сетка гиперпараметров ===
prophet_grid = {
    'n_changepoints': [50, 100, 150],
    'changepoint_prior_scale': [0.5, 2.0, 5.0]
}
lgbm_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.03, 0.1],
    'num_leaves': [31, 64, 128]
}

best_score = np.inf
best_params = {}
ts_split = TimeSeriesSplit(n_splits=3)

# Перебор комбинаций Prophet и GridSearch для LGBM
for cps in prophet_grid['n_changepoints']:
    for cpscale in prophet_grid['changepoint_prior_scale']:
        # Prophet
        m = Prophet(
            yearly_seasonality=True,
            n_changepoints=cps,
            changepoint_prior_scale=cpscale,
            holidays=crisis_events,
            holidays_prior_scale=1.0
        )
        m.add_seasonality('monthly', 30.5, fourier_order=5)
        m.add_seasonality('quarterly', 91.25, fourier_order=3)
        for feat in ['Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]:
            m.add_regressor(feat)
        m.fit(prophet_train)

        # Генерация фич
        all_future = pd.concat([prophet_train, prophet_test], axis=0).reset_index(drop=True)
        fc = m.predict(all_future)
        ft = fc.iloc[len(prophet_train):]

        df_lgbm = (
            prepare(df_test)[['ds','y'] + ['Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]]
            .merge(ft[['ds','yhat','trend']], on='ds')
        )
        X = df_lgbm[['yhat','trend','Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]]
        y = df_lgbm['y']

        # GridSearchCV для LGBM
        grid = GridSearchCV(
            LGBMRegressor(random_state=42),
            param_grid=lgbm_grid,
            cv=ts_split,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        grid.fit(X, y)
        score = -grid.best_score_

        if score < best_score:
            best_score = score
            best_params = {
                'prophet': {'n_changepoints': cps, 'changepoint_prior_scale': cpscale},
                'lgbm': grid.best_params_
            }

print('Лучший RMSE:', best_score)
print('Лучшие параметры:', best_params)

# === 5. Финальное обучение и оценка ===
bp = best_params
# Prophet финальный
model = Prophet(
    yearly_seasonality=True,
    n_changepoints=bp['prophet']['n_changepoints'],
    changepoint_prior_scale=bp['prophet']['changepoint_prior_scale'],
    holidays=crisis_events,
    holidays_prior_scale=1.0
)
model.add_seasonality('monthly', 30.5, fourier_order=5)
model.add_seasonality('quarterly', 91.25, fourier_order=3)
for feat in ['Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]:
    model.add_regressor(feat)
model.fit(prophet_train)

# Генерация прогноза для теста
all_future = pd.concat([prophet_train, prophet_test], axis=0).reset_index(drop=True)
forecast = model.predict(all_future)
forecast_test = forecast.iloc[len(prophet_train):]

# Подготовка финальных LGBM-фич
df_final = (
    prepare(df_test)[['ds','y'] + ['Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]]
    .merge(forecast_test[['ds','yhat','trend']], on='ds')
)
X_final = df_final[['yhat','trend','Oil_Lag1','Oil_SMA_3','Oil_Lag6','Oil_Lag12'] + [f'is_{g}' for g in groups]]
y_final = df_final['y']

lgbm_final = LGBMRegressor(**bp['lgbm'], random_state=42)
lgbm_final.fit(X_final, y_final)

# Оценка
y_pred = lgbm_final.predict(X_final)
y_true = y_final

y_true_exp = np.expm1(y_true)
y_pred_exp = np.expm1(y_pred)
print('KPP =', KPP(y_true_exp, y_pred_exp)*100)
print('KPP_tan =', KPP_tan.KPP(y_true_exp, y_pred_exp))
print('MAE =', mean_absolute_error(y_true_exp, y_pred_exp))
print('RMSE =', np.sqrt(mean_squared_error(y_true_exp, y_pred_exp)))
print('MAPE =', mean_absolute_percentage_error(y_true_exp, y_pred_exp))

# Визуализация
plt.figure(figsize=(14,6))
plt.plot(df_train['Date'], df_train['Freight_Price'], label='Train')
plt.plot(df_test['Date'], df_test['Freight_Price'], label='Test')
plt.plot(df_test['Date'], y_pred_exp, '--', label='Forecast')
plt.legend()
plt.xticks(rotation=45)
plt.title('Прогноз с Grid Search гиперпараметров')
plt.tight_layout()
plt.show()
