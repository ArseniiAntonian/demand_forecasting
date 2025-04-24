import pandas as pd
import numpy as np
from itertools import combinations
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# === Загрузка и подготовка ===
df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/ML.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'])
df['Freight_Lag24'] = df['Freight_Price'].shift(24)
df['Oil_Lag1'] = df['Oil_Price'].shift(1)
df['Oil_Lag2'] = df['Oil_Price'].shift(2)
df['Oil_SMA_3'] = df['Oil_Price'].rolling(3).mean()
df['Log_Freight'] = np.log1p(df['Freight_Price'])

df = df.dropna().reset_index(drop=True)

# === Train/Test split ===
train = df[df['Date'] < '2023-01-03'].copy()
test = df[(df['Date'] >= '2023-01-03') & (df['Date'] < '2025-01-03')].copy()

# === Потенциальные признаки ===
base_features = [
    'Oil_Price', 'Oil_Lag1', 'Oil_Lag2', 'Oil_SMA_3',
    #'crisis_Inflation and rate hikes',
    'has_crisis', 'has_war', 'has_pandemic',
    'Freight_Lag24'
]

# Генерация комбинаций фичей
feature_sets = [list(comb) for i in range(3, 7) for comb in combinations(base_features, i)]
changepoint_values = [0.01, 0.1, 0.5]

# === Функция подготовки данных ===
def prepare_df(df_part, features):
    df_model = df_part[['Date', 'Log_Freight'] + features].copy()
    df_model = df_model.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})
    return df_model

# === Grid Search ===
results = []

for features in feature_sets:
    for cp in changepoint_values:
        try:
            freight_train = prepare_df(train, features)
            future = prepare_df(test, features)

            model = Prophet(yearly_seasonality=True, changepoint_prior_scale=cp)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
            for f in features:
                model.add_regressor(f)

            model.fit(freight_train)
            forecast = model.predict(future)

            y_true = test['Freight_Price'].values
            y_pred = np.expm1(forecast['yhat'].values)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))


            results.append({
                'features': features,
                'changepoint_prior_scale': cp,
                'mae': mae,
                'rmse': rmse
            })

        except Exception as e:
            print(f"Ошибка с фичами {features} и cp={cp}: {e}")
            continue

# === Вывод топ-результатов ===
results_df = pd.DataFrame(results).sort_values(by='mae').reset_index(drop=True)
print("\nТОП-10 лучших конфигураций по MAE:")
print(results_df.head(10))
