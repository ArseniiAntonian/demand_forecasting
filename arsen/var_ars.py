import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Загрузка и предварительная обработка данных
oil_prices = pd.read_csv("data/monthly_oil_cost_1988-2025.csv", usecols=["Date", "Price"])
freight_prices = pd.read_csv("data/cleanFreight.csv", usecols=["Date", "Price"])

oil_prices["Price"] = oil_prices["Price"].astype(float)
freight_prices["Price"] = freight_prices["Price"].replace(',', '', regex=True).astype(float)

oil_prices["Date"] = pd.to_datetime(oil_prices["Date"], format="%m/%d/%Y")
freight_prices["Date"] = pd.to_datetime(freight_prices["Date"], format="%m/%d/%Y")

df = pd.merge(freight_prices, oil_prices, on="Date", suffixes=("_freight", "_oil"))
df.sort_values(by="Date", inplace=True)
df.set_index("Date", inplace=True)
df.index.freq = 'MS'

# 1.1. Winsorization для устранения выбросов
def winsorize_series(series, lower_quantile=0.05, upper_quantile=0.95):
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower, upper)

df['Price_freight'] = winsorize_series(df['Price_freight'])
df['Price_oil'] = winsorize_series(df['Price_oil'])

# Визуализация после winsorization
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Price_freight'], label='Цена фрахта', color='blue')
plt.title('Цена фрахта после Winsorization')
plt.subplot(2, 1, 2)
plt.plot(df.index, df['Price_oil'], label='Цена нефти', color='orange')
plt.title('Цена нефти после Winsorization')
plt.tight_layout()
plt.show()

# 1.2. Логарифмирование для стабилизации дисперсии
df['Log_Price_freight'] = np.log(df['Price_freight'])
df['Log_Price_oil'] = np.log(df['Price_oil'])
df_log = df[['Log_Price_freight', 'Log_Price_oil']]

# 2. Сезонное дифференцирование (лаг=12)
df_diff = df_log.diff(12).dropna()

# 3. Разделение на обучающую и тестовую выборки (80%/20%)
train_size = int(len(df_diff) * 0.8)
train = df_diff.iloc[:train_size]
test = df_diff.iloc[train_size:]

# 4. Построение моделей

# 4.1. VARMAX для обоих рядов (порядок (3,0) выбран как пример)
model_varmax = VARMAX(train, order=(3, 0))
results_varmax = model_varmax.fit(disp=False)
print("VARMAX Model Summary:")
print(results_varmax.summary())

# Прогноз VARMAX на тестовый период
forecast_steps = len(test)
forecast_varmax_diff = results_varmax.get_forecast(steps=forecast_steps).predicted_mean
forecast_varmax_diff.index = test.index

# 4.2. SARIMAX для цены фрахта с ценой нефти как экзогенной переменной
model_sarimax = SARIMAX(train['Log_Price_freight'],
                        exog=train['Log_Price_oil'],
                        order=(1, 0, 1),
                        seasonal_order=(1, 0, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
results_sarimax = model_sarimax.fit(disp=False)
print("SARIMAX Model Summary:")
print(results_sarimax.summary())

# Прогноз SARIMAX на тестовом периоде (с использованием экзогенной переменной из теста)
forecast_sarimax_diff = results_sarimax.get_forecast(steps=forecast_steps, exog=test['Log_Price_oil']).predicted_mean
forecast_sarimax_diff.index = test.index

# 5. Итеративная функция для обратного сезонного дифференцирования
def iterative_invert_seasonal_diff(forecast_diff, train_actual, season=12):
    """
    Итеративно инвертирует сезонное дифференцирование.
    forecast_diff: Series с прогнозируемыми разностями (на логарифмированных данных)
    train_actual: Исходный логарифмированный ряд (например, df_log['Log_Price_freight'])
                  до конца обучающей выборки.
    season: сезонный лаг (по умолчанию 12)
    """
    # Инициализируем history последними season значениями из обучающего набора
    history = list(train_actual[-season:])
    forecast_inverted = []
    for i, t in enumerate(forecast_diff.index):
        if i < season:
            base = history[i]
        else:
            base = forecast_inverted[i - season]
        forecast_inverted.append(forecast_diff.loc[t] + base)
    return pd.Series(forecast_inverted, index=forecast_diff.index)

# 6. Обратное преобразование прогнозов
# 6.1. Для VARMAX (для обоих рядов)
forecast_varmax_inverted = forecast_varmax_diff.copy()

# Для каждого столбца (каждого ряда) применяем итеративное инвертирование
for col in forecast_varmax_diff.columns:
    forecast_varmax_inverted[col] = iterative_invert_seasonal_diff(forecast_varmax_diff[col],
                                                                     df_log[col].loc[:train.index[-1]],
                                                                     season=12)
# Преобразуем в исходный масштаб (экспоненцирование)
forecast_varmax_final = np.exp(forecast_varmax_inverted)
forecast_varmax_final.rename(columns={"Log_Price_freight": "Freight", "Log_Price_oil": "Oil"}, inplace=True)

# 6.2. Для SARIMAX (только для цены фрахта)
forecast_sarimax_inverted = iterative_invert_seasonal_diff(forecast_sarimax_diff,
                                                           df_log['Log_Price_freight'].loc[:train.index[-1]],
                                                           season=12)
forecast_sarimax_final = np.exp(forecast_sarimax_inverted)

# 7. Восстановление истинных значений цены фрахта для тестового периода
# Для теста вычисляем восстановленное логарифмическое значение, прибавляя фактическое значение за тот же месяц предыдущего года
test_actual_log = []
for t in test.index:
    previous_time = t - pd.DateOffset(months=12)
    if previous_time in df_log.index:
        base_val = df_log.loc[previous_time, 'Log_Price_freight']
    else:
        base_val = df_log['Log_Price_freight'].iloc[-12]
    test_actual_log.append(test.loc[t, 'Log_Price_freight'] + base_val)
test_actual_log = pd.Series(test_actual_log, index=test.index)
test_actual_final = np.exp(test_actual_log)

# 8. Оценка точности прогнозов (только для цены фрахта)
rmse_varmax = np.sqrt(mean_squared_error(test_actual_final, forecast_varmax_final["Freight"]))
mae_varmax = mean_absolute_error(test_actual_final, forecast_varmax_final["Freight"])

rmse_sarimax = np.sqrt(mean_squared_error(test_actual_final, forecast_sarimax_final))
mae_sarimax = mean_absolute_error(test_actual_final, forecast_sarimax_final)

print("\nForecast Accuracy for Freight Prices:")
print("VARMAX - RMSE: {:.4f}, MAE: {:.4f}".format(rmse_varmax, mae_varmax))
print("SARIMAX - RMSE: {:.4f}, MAE: {:.4f}".format(rmse_sarimax, mae_sarimax))

# 9. Построение финального графика, где историческая цена фрахта (train + test) выводится полностью
# Используем исходный ряд (после winsorization) с момента, когда доступны данные для инверсии (начиная с 13-й точки)
hist_start = df_log.index[12]  # первые 12 точек не доступны после дифференцирования
hist_freight = df.loc[hist_start:, "Price_freight"]

plt.figure(figsize=(12, 6))
plt.plot(hist_freight.index, hist_freight, label="Историческая цена фрахта", color="blue")
plt.plot(forecast_varmax_final.index, forecast_varmax_final["Freight"], label="VARMAX Прогноз", color="red", linestyle="--")
plt.plot(forecast_sarimax_final.index, forecast_sarimax_final, label="SARIMAX Прогноз", color="orange", linestyle="--")
plt.title("Сравнение прогнозов цены фрахта (Исторические данные (Train + Test) и прогнозы)")
plt.xlabel("Дата")
plt.ylabel("Цена фрахта")
plt.legend()
plt.grid(True)
plt.show()
