import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, ReLU, Dropout, GlobalAveragePooling1D
from data_prep import get_data

# === 1. Загрузка данных ===
data = np.load('data/tcn_dataset.npz')
X_train, y_train, X_test, y_test, freight_idx = get_data(12)
print(X_train.shap)

# === 2. Построение модели TCN ===
input_shape = X_train.shape[1:]
output_dim = y_train.shape[1]

model = Sequential([
    Input(shape=input_shape),
    Conv1D(64, kernel_size=3, padding="causal", dilation_rate=1),
    BatchNormalization(),
    ReLU(),
    Dropout(0.2),
    
    Conv1D(64, kernel_size=3, padding="causal", dilation_rate=2),
    BatchNormalization(),
    ReLU(),
    Dropout(0.2),
    
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dense(output_dim)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# === 3. Обучение ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

# === 4. Оценка ===
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# === 5. Прогноз ===
y_pred = model.predict(X_test)
print(y_pred.shape)

# === 6. Визуализация одного окна: вход + предсказания + факты ===

# Индекс окна, который хочешь посмотреть
window_index = 0

# Вытаскиваем последнее значение Freight_Price из каждого временного шага
freight_window = X_test[window_index, :, freight_idx]
true_future = y_test[window_index]
pred_future = y_pred[window_index]

# Объединяем всё в один временной ряд
full_true = np.concatenate([freight_window, true_future])
full_pred = np.concatenate([freight_window, pred_future])
timesteps = np.arange(len(full_true))

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(timesteps, full_true, label='Истинные значения', color='black')
plt.plot(timesteps, full_pred, '--', label='Предсказания модели', color='tab:blue')
plt.axvline(x=len(freight_window) - 1, color='gray', linestyle=':', label='Граница прогноза')
plt.title(f"TCN: вход + прогноз для тестового окна #{window_index}")
plt.xlabel("Месяцы (относительно начала окна)")
plt.ylabel("Freight Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()