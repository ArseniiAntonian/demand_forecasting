import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN  # pip install keras-tcn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    # Загружаем данные (замените путь на ваш реальный путь к файлам)
    oil_prices = pd.read_csv("data/monthly_oil_cost_1988-2025.csv")  # Датасет цен на нефть
    oil_prices = oil_prices.drop(['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1)
    freight_prices = pd.read_csv("data/freight_cost.csv")  # Датасет цен на фрахт
    freight_prices = freight_prices.drop(['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1)

    # Преобразуем дату в формат datetime
    oil_prices["Date"] = pd.to_datetime(oil_prices["Date"], format="%m/%d/%Y")
    freight_prices["Date"] = pd.to_datetime(freight_prices["Date"], format="%m/%d/%Y")

    # Объединяем два датасета по дате
    df = pd.merge(oil_prices, freight_prices, on="Date", suffixes=("_oil", "_freight"))

    # Добавляем лаговые признаки (цены на нефть и фрахт за месяц назад)
    df["oil_price_lag1"] = df["Price_oil"].shift(1)
    df["freight_price_lag1"] = df["Price_freight"].shift(1)

    # Добавляем сезонность (номер месяца)
    df["month"] = df["Date"].dt.month

    # Удаляем строки с пропущенными значениями (из-за лагов)
    df.dropna(inplace=True)
    return df

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 5))
    plt.scatter(range(len(residuals)), residuals, color='purple', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='dashed')
    plt.xlabel("Index")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

def plot_loss(history):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

df = load_data()
df["Crisis"] = (df["Date"] >= pd.to_datetime("2008-01-01")).astype(int) - 1
df["Crisis_x_OilPrice"] = df["Crisis"] * df["Price_oil"]
df["Crisis_x_FreightPrice"] = df["Crisis"] * df["Price_freight"]
df["OilPrice_lag2"] = df["Price_oil"].shift(2)
df["FreightPrice_lag2"] = df["Price_freight"].shift(2)
df["Smoothed_OilPrice"] = df["Price_oil"].ewm(span=6, adjust=False).mean()
df["Smoothed_FreightPrice"] = df["Price_freight"].ewm(span=6, adjust=False).mean()
df["sin_month"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
df["cos_month"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)

# Добавляем лаги за 3 и 6 месяцев
df["OilPrice_lag3"] = df["Price_oil"].shift(3)
df["FreightPrice_lag3"] = df["Price_freight"].shift(3)
df["OilPrice_lag6"] = df["Price_oil"].shift(6)
df["FreightPrice_lag6"] = df["Price_freight"].shift(6)
df.dropna(inplace=True)
print(df.tail(12))


features = ['Price_oil', 'oil_price_lag1', 'freight_price_lag1', 'Crisis', 'Smoothed_OilPrice', 'Crisis_x_OilPrice', 'Crisis_x_FreightPrice',
             'Smoothed_FreightPrice', 'sin_month', 'cos_month', 'OilPrice_lag3', 'FreightPrice_lag3', 'OilPrice_lag6', 'FreightPrice_lag6']
target = 'Price_freight'

X = df[features].values
y = df[target].values

# Масштабируем данные
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Преобразуем в формат (samples, timesteps, features) для TCN
X = X.reshape(X.shape[0], 1, X.shape[1])

# Разделяем на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# Создаём TCN модель
model = Sequential([
    TCN(input_shape=(1, X.shape[2]),
        nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8],
        return_sequences=False),
    Dense(1)
])

# model = Sequential([
#     TCN(input_shape=(1, X.shape[2]),
#         nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8],
#         dropout_rate = 0.3, return_sequences=False),
#     Dense(1)
# ])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Обучаем модель
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Делаем предсказание
predictions = model.predict(X_test)

# Обратно масштабируем
predictions = scaler_y.inverse_transform(predictions)
y_test_original = scaler_y.inverse_transform(y_test)

# Визуализируем прогноз
plt.figure(figsize=(12, 5))
plt.plot(y_test_original, label="Real Freight Price", color='blue')
plt.plot(predictions, label="Predicted Freight Price", color='red', linestyle='dashed')
plt.legend()
plt.title("Freight Price Prediction using TCN")
plt.show()

# Вызов функции для визуализации остатков
plot_residuals(y_test_original, predictions)
plot_loss(history)
