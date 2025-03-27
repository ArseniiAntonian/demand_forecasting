import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K

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
    df["target"] = df["Price_freight"].shift(-2)  # Сдвигаем цель на 2 месяца вперёд
    return df

def plot_loss(history):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

def analyze_and_correct_bias(y_true, y_pred):
    bias = (y_pred - y_true).mean()
    print(f"Среднее смещение предсказания: {bias:.2f}")
    corrected_pred = y_pred - bias
    return corrected_pred

def time_shifted_mse(y_true, y_pred):
    y_true_shifted = K.concatenate([y_true[:, 2:], y_true[:, :2]], axis=1)  # Смещаем на 2 шага вперёд
    return K.mean(K.square(y_true_shifted - y_pred))

# Загрузка и обработка данных
df = load_data()
df.dropna(inplace=True)

# Добавляем дополнительные лаги
df["Crisis"] = (df["Date"] >= pd.to_datetime("2008-01-01")).astype(int)
df["Crisis_x_OilPrice"] = df["Crisis"] * df["Price_oil"]
df["Crisis_x_FreightPrice"] = df["Crisis"] * df["Price_freight"]

for lag in [2, 3, 4, 5, 6, 9, 12, 18, 24]:
    df[f"OilPrice_lag{lag}"] = df["Price_oil"].shift(lag)
    df[f"FreightPrice_lag{lag}"] = df["Price_freight"].shift(lag)

df["RollingMean_OilPrice"] = df["Price_oil"].rolling(window=6).mean()
df["RollingMean_FreightPrice"] = df["Price_freight"].rolling(window=3).mean()
df["RollingStd_OilPrice"] = df["Price_oil"].rolling(window=6).std()
df["RollingStd_FreightPrice"] = df["Price_freight"].rolling(window=3).std()
df["Momentum_Oil"] = df["Price_oil"] - df["RollingMean_OilPrice"]
df["Momentum_Freight"] = df["Price_freight"] - df["RollingMean_FreightPrice"]
df["Oil_x_Freight"] = df["Price_oil"] * df["Price_freight"]
df["Lag1_Oil_x_Lag1_Freight"] = df["oil_price_lag1"] * df["freight_price_lag1"]
df["Crisis_x_OilChange"] = df["Crisis"] * (df["Price_oil"] - df["oil_price_lag1"])
df["Crisis_x_FreightChange"] = df["Crisis"] * (df["Price_freight"] - df["freight_price_lag1"])
df["sin_month"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
df["cos_month"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
df.dropna(inplace=True)

features = list(df.columns.difference(["Date", "target"]))
target = 'target'

X = df[features].values
y = df[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X = X.reshape(X.shape[0], 1, X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

model = Sequential([
    TCN(input_shape=(1, X.shape[2]),
        nb_filters=64,
        kernel_size=5,  # Расширяем окно восприятия модели
        dilations=[1, 2, 4, 8, 16, 32],  # Увеличиваем дилатацию для дальновидных зависимостей
        dropout_rate=0.2,
        return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss=time_shifted_mse)
model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

predictions = model.predict(X_test)
predictions = scaler_y.inverse_transform(predictions)
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))
corrected_predictions = analyze_and_correct_bias(y_test_original, predictions)

y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1))

# Финальный график: тренировка + тест + предсказание
plt.figure(figsize=(14, 6))
plt.plot(np.arange(len(y_train_original)), y_train_original, label="Train", color='blue')
plt.plot(np.arange(len(y_train_original), len(y_train_original) + len(y_test_original)),
         y_test_original, label="Real Test", color='orange')
plt.plot(np.arange(len(y_train_original), len(y_train_original) + len(predictions)),
         predictions, label="Raw Prediction", color='red', linestyle='dashed')
plt.plot(np.arange(len(y_train_original), len(y_train_original) + len(corrected_predictions)),
         corrected_predictions, label="Bias-Corrected Prediction", color='green', linestyle='dotted')
plt.axvline(x=len(y_train_original)-1, color='gray', linestyle='--', label='Train/Test Split')
plt.title("Freight Price Prediction using TCN with Time-Shifted MSE")
plt.xlabel("Time Index")
plt.ylabel("Freight Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plot_loss(history)
