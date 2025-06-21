import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error  # для RMSE, так как sklearn.metrics.root_mean_squared_error появляется только в sklearn >=1.3
from scipy.stats import kendalltau, theilslopes
import KPP_metrik_trend as KPP_metrik_trend
import KPP_tan
import pandas as pd
import numpy as np
from prophet import Prophet

# === Загрузка и предобработка данных ===
df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/ML.csv', parse_dates=['Date'])

# === Преобразование даты и создание лагов/скользящей средней ===
df['Date'] = pd.to_datetime(df['Date'])
df['Oil_Lag1'] = df['Oil_Price'].shift(1)
df['Oil_Lag2'] = df['Oil_Price'].shift(2)
df['Oil_SMA_3'] = df['Oil_Price'].rolling(3).mean()
df['Oil_Lag6'] = df['Oil_Price'].shift(6)
df['Oil_Lag12'] = df['Oil_Price'].shift(12)
df['Oil_Change_1m'] = df['Oil_Price'].pct_change(periods=1)

# === Логарифм целевой переменной ===
df['Log_Freight'] = np.log1p(df['Freight_Price'])

# === Удаляем строки с пропущенными значениями ===
df = df.dropna().reset_index(drop=True)

# === Разделение на train/test ===
train = df[df['Date'] < '2020-01-03'].copy()
test = df[(df['Date'] >= '2020-01-03') & (df['Date'] <= '2025-01-03')].copy()

used_features = [
    'Oil_Lag1', 
    'Oil_SMA_3',
    'Oil_Lag6',
    'Oil_Lag12'
]

# === Подготовка датафрейма для Prophet ===
def prepare_df(df_part):
    df_model = df_part[['Date', 'Log_Freight'] + used_features].copy()
    df_model = df_model.rename(columns={'Date': 'ds', 'Log_Freight': 'y'})
    return df_model

freight_train = prepare_df(train)
future = prepare_df(test)

# === Добавим пользовательские события (holidays/критические события) ===
crisis_events = pd.DataFrame({
    'holiday': [
        'covid_lockdown', 
        'suez_blockage', 
        'ukraine_conflict', 
        'china_port_closure',
        'inflation_peak'
    ],
    'ds': pd.to_datetime([
        '2020-03-15', 
        '2021-03-23', 
        '2022-02-24', 
        '2021-08-01', 
        '2022-06-01'
    ]),
    'lower_window': 0,
    'upper_window': 180
})

# === Обучение модели Prophet ===
model = Prophet(
    yearly_seasonality=True,
    changepoint_prior_scale=2,
    holidays=crisis_events
)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

for feature in used_features:
    model.add_regressor(feature)

model.fit(freight_train)

# === Прогноз ===
forecast = model.predict(future)

# Объединяем прогноз с тестовой частью
test = test.merge(forecast[['ds', 'yhat', 'trend']], left_on='Date', right_on='ds', how='left')
test['yhat_exp'] = np.expm1(test['yhat'])
print('KPP метрика = ', KPP_metrik_trend.KPP(test)*100, ' %')
print('KPP_tan метрика = ', KPP_tan.KPP(test))
# === Визуализация ===
plt.figure(figsize=(14, 6))
plt.plot(train['Date'], train['Freight_Price'], label='Тренировочные данные', color='blue')
plt.plot(test['Date'], test['Freight_Price'], label='Тест (2020–2025)', color='blue')
plt.plot(test['Date'], test['yhat_exp'], '--', label='Прогноз', color='red')

plt.title("Прогноз фрахта: Prophet + внешние признаки (2020–2025)")
plt.xlabel("Дата")
plt.ylabel("Фрахтовая ставка")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Метрики: MAE и RMSE ===
mae = mean_absolute_error(test['Freight_Price'], test['yhat_exp'])
rmse = np.sqrt(mean_squared_error(test['Freight_Price'], test['yhat_exp']))  # root mean squared error

print("----- Test (2020-2025) -----")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# === 1) Отклонение тренда (разница Sen’s Slope actual vs predicted) ===
# Для оценки наклона используем theilslopes из scipy.stats.
# Берём в качестве x числовое представление дат (ordinal),
# а в качестве y — реальные и предсказанные значения Freight_Price.

# Переводим даты в ordinal (целое число) для регрессии
# x_ord = test['Date'].map(pd.Timestamp.toordinal).values

# # Sen’s slope для тестовых фактических значений
# test_slope, test_intercept, test_low, test_up = theilslopes(
#     test['Freight_Price'].values, 
#     x_ord, 
#     0.95
# )

# # Sen’s slope для прогнозных значений
# pred_slope, pred_intercept, pred_low, pred_up = theilslopes(
#     test['yhat_exp'].values, 
#     x_ord, 
#     0.95
# )

# slope_diff = pred_slope - test_slope

# print("\n--- Оценка наклона (Sen’s Slope) ---")
# print(f"  Sen’s Slope (Test actual)      : {test_slope:.6f}")
# print(f"  Sen’s Slope (Test predicted)   : {pred_slope:.6f}")
# print(f"  Разница (predicted − actual)   : {slope_diff:.6f}")

# # === 2) Коэффициент ранговой корреляции τ Манна–Кендалла ===
# tau, p_value = kendalltau(test['Freight_Price'], test['yhat_exp'])
# print("\n--- Коэффициент τ Манна–Кендалла ---")
# print(f"  τ   : {tau:.4f}")
# print(f"  p-value : {p_value:.4f}")

# # === 3) Отдельная оценка наклона Sen’s Slope (для трендов теста и предикта) ===
# # (вроде пункта 1, но можно показать отдельный блок, если нужно)
# print("\n--- Sen’s Slope (Theil–Sen) для сравнения трендов ---")
# print(f"  Тестовый ряд: наклон = {test_slope:.6f}")
# print(f"  Прогноз      : наклон = {pred_slope:.6f}")
# print(f"  Разница в наклоне = {slope_diff:.6f}")
