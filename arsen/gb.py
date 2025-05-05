def forecast_lgbе(data):
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt

    # Загрузка и предобработка данных
    df = pd.read_csv('data/FullDataset.csv')
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
    lgb1 = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=5, random_state=42)
    lgb1.fit(X_train, y_train)
    y_pred_lgb = lgb1.predict(X_test)
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    mse_lgb = mean_squared_error(y_test, y_pred_lgb)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    print(f"LightGBM Regressor - MAE: {mae_lgb}, MSE: {mse_lgb}, R2: {r2_lgb}")
    return y_train, y_test, y_pred_lgb

# Построение графика
# plt.figure(figsize=(14, 7))
# plt.plot(y.index, y, label='Actual', color='blue')
# plt.plot(y_test.index, y_pred_lgb, label='Predicted', color='red', linestyle='--')
# plt.title('Предсказания LightGBM модели')
# plt.xlabel('Дата')
# plt.ylabel('Цена фрахта')
# plt.legend()
# plt.grid(True)
# plt.show()