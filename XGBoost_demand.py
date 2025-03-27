import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

def load_data():
    oil_prices = pd.read_csv("data/monthly_oil_cost_1988-2025.csv")
    oil_prices = oil_prices.drop(['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1)
    oil_prices['Price'] = oil_prices['Price'].astype(float)
    freight_prices = pd.read_csv("data/cleanFreight.csv")
    freight_prices = freight_prices.drop(['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1)
    freight_prices['Price'] = freight_prices['Price'].str.replace(',', '').astype(float)

    oil_prices["Date"] = pd.to_datetime(oil_prices["Date"], format="%m/%d/%Y")
    freight_prices["Date"] = pd.to_datetime(freight_prices["Date"], format="%m/%d/%Y")

    df = pd.merge(oil_prices, freight_prices, on="Date", suffixes=("_oil", "_freight"))
    df["oil_price_lag1"] = df["Price_oil"].shift(1)
    df["freight_price_lag1"] = df["Price_freight"].shift(1)
    df["month"] = df["Date"].dt.month
    df["target"] = df["Price_freight"].shift(-1)
    return df

# Загрузка и обработка данных
df = load_data()
df["Crisis"] = (df["Date"] >= pd.to_datetime("2008-01-01")).astype(int)
df["Crisis_x_OilPrice"] = df["Crisis"] * df["Price_oil"]
df["Crisis_x_FreightPrice"] = df["Crisis"] * df["Price_freight"]

for lag in [2, 3, 4, 5, 6, 9, 12, 18, 24]:
    df[f"OilPrice_lag{lag}"] = df["Price_oil"].shift(lag)
    df[f"FreightPrice_lag{lag}"] = df["Price_freight"].shift(lag)

df["Smoothed_OilPrice"] = df["Price_oil"].ewm(span=6, adjust=False).mean()
df["Smoothed_FreightPrice"] = df["Price_freight"].ewm(span=6, adjust=False).mean()
df["sin_month"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
df["cos_month"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
df["RollingMean_OilPrice"] = df["Price_oil"].rolling(window=6).mean()
df["RollingMean_FreightPrice"] = df["Price_freight"].rolling(window=3).mean()
df.dropna(inplace=True)

features = [
    'Price_oil', 'oil_price_lag1', 'freight_price_lag1', 'Crisis'
] + [
    f"OilPrice_lag{lag}" for lag in [2, 3, 4, 5, 6, 9, 12, 18, 24]
] + [
    f"FreightPrice_lag{lag}" for lag in [2, 3, 4, 5, 6, 9, 12, 18, 24]
]
target = 'target'

X = df[features].values
y = df[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# Обучаем XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

# Предсказания
predictions = model.predict(X_test)
y_pred_original = scaler_y.inverse_transform(predictions.reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# График
plt.figure(figsize=(14, 6))
plt.plot(np.arange(len(y_train)), scaler_y.inverse_transform(y_train.reshape(-1, 1)), label="Train", color='blue')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_original, label="Real Test", color='orange')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_original, label="XGBoost Prediction", color='green', linestyle='dotted')
plt.axvline(x=len(y_train)-1, color='gray', linestyle='--', label='Train/Test Split')
plt.title("Freight Price Prediction using XGBoost")
plt.xlabel("Time Index")
plt.ylabel("Freight Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Метрики
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
