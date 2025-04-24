import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Загрузка и предобработка данных
df = pd.read_csv('../data/FullDataset.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df['Month'] = df['Date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df.sort_values(by="Date", inplace=True)
df.set_index("Date", inplace=True)
df.index.freq = 'MS'

# Создание отстающих признаков для LightGBM
df['Freight_Price(t-1)'] = df['Freight_Price'].shift(1)
df['Freight_Price(t-6)'] = df['Freight_Price'].shift(6)
df['Freight_Price(t-12)'] = df['Freight_Price'].shift(12)
df['Freight_Price(t-24)'] = df['Freight_Price'].shift(24)
df['Oil_Price(t-1)'] = df['Oil_Price'].shift(1)
df['Oil_Price(t-6)'] = df['Oil_Price'].shift(6)
df['Oil_Price(t-12)'] = df['Oil_Price'].shift(12)
df['Oil_Price(t-24)'] = df['Oil_Price'].shift(24)
df = df.dropna()

X = df.drop(columns=['Freight_Price'])
y = df['Freight_Price']

# Разделение данных (последние 24 записи для теста)
X_train, X_test = X[:-24], X[-24:]
y_train, y_test = y[:-24], y[-24:]

# Обучение базовой модели LightGBM
lgb1 = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgb1.fit(X_train, y_train)
y_pred_lgb = lgb1.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f"LightGBM Regressor - MAE: {mae_lgb}, MSE: {mse_lgb}, R2: {r2_lgb}")

# Подбор гиперпараметров для LightGBM
lgb_param_grid = {
    'n_estimators': [25, 50, 100, 200, 300],
    'learning_rate': [0.001, 0.01, 0.015, 0.025, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 10],
    'subsample': [0.8, 1.0]
}
lgb2 = lgb.LGBMRegressor(random_state=42)
grid_search_lgb = GridSearchCV(estimator=lgb2,
                               param_grid=lgb_param_grid,
                               scoring='neg_mean_absolute_error',
                               n_jobs=-1)
grid_search_lgb.fit(X_train, y_train)
print(f"Лучшие гиперпараметры LightGBM: {grid_search_lgb.best_params_}")
best_lgb = grid_search_lgb.best_estimator_
y_pred_best_lgb = best_lgb.predict(X_test)
mae_best_lgb = mean_absolute_error(y_test, y_pred_best_lgb)
mse_best_lgb = mean_squared_error(y_test, y_pred_best_lgb)
r2_best_lgb = r2_score(y_test, y_pred_best_lgb)
print(f"LightGBM с лучшими гиперпараметрами - MAE: {mae_best_lgb}, MSE: {mse_best_lgb}, R2: {r2_best_lgb}")

# Построение графика
plt.figure(figsize=(14, 7))
plt.plot(y.index, y, label='Actual', color='blue')
plt.plot(y_test.index, y_pred_best_lgb, label='Predicted', color='red', linestyle='--')
plt.title('Предсказания LightGBM модели')
plt.xlabel('Дата')
plt.ylabel('Цена фрахта')
plt.legend()
plt.grid(True)
plt.show()