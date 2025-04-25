import matplotlib.pyplot as plt
def forecast_seq2seq(data):   # TODO: Обработка входных кризисов 
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, RepeatVector, TimeDistributed
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math

    if tf.config.list_physical_devices('GPU'):
        print("Using GPU")
    else:
        print("Using CPU")

    # Импортируем TCN из keras-tcn
    from tcn import TCN

    # 1. Загрузка и предварительная обработка данных
    df = pd.read_csv("data/ML.csv", parse_dates=["Date"])
    df['Oil_Lag6'] = df["Oil_Lag2"].shift(4)
    df['Freight_Lag6'] = df["Freight_Lag2"].shift(4)
    df.dropna(inplace=True)
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    # Определяем списки признаков
    time_series_cols = [
        "Freight_Price",  "Oil_Price",
        "Freight_Lag1", "Freight_Lag2",
        "Oil_Lag1", "Oil_Lag2",
        "Oil_Lag6", "Freight_Lag6",
        # Можно добавить и другие временные признаки, если требуется.
    ]

    categorical_cols = [
        # "month", "year", "quarter", 
        "is_year_start", "is_year_end",
        # "sin_month", "cos_month",
        "has_crisis", "has_war", "has_sanctions", "has_pandemic"
    ]

    combined_cols = time_series_cols + categorical_cols
    df = df[combined_cols].copy()
    df.dropna(inplace=True)

    print("Первые строки данных:")
    print(df.head(5))

    # 2. Нормализация данных
    # Нормализуем все входные признаки
    scaler_X = MinMaxScaler()
    X_categorical = df[categorical_cols].values  # shape: (n_samples, n_categorical_features)
    X_scaled = scaler_X.fit_transform(df[time_series_cols]) 
    X_scaled = np.concatenate((X_scaled, X_categorical), axis=1)
    X_df = pd.DataFrame(X_scaled, columns=combined_cols)
    print(X_df.head(5))
    n_features = X_scaled.shape[1]  # число признаков для энкодера

    # Нормализуем целевую переменную Freight_Price отдельно
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(df[['Freight_Price']])  # shape: (n_samples, 1)

    # 3. Формирование последовательностей для обучения
    def create_sequences(X_data, y_data, n_input, n_output):
        """
        Формирует обучающие примеры:
        - encoder_X: входная последовательность для энкодера, форма (n_input, n_features)
        - decoder_y: целевая последовательность для предсказания Freight_Price, форма (n_output, 1)
        """
        encoder_X, decoder_y = [], []
        for i in range(len(X_data) - n_input - n_output):
            encoder_seq = X_data[i : i + n_input]
            target_seq = y_data[i + n_input : i + n_input + n_output]
            encoder_X.append(encoder_seq)
            decoder_y.append(target_seq)
        return np.array(encoder_X), np.array(decoder_y)

    # Выбираем гиперпараметры: длина входной последовательности и горизонт прогноза
    n_input = 18   # например, 12 временных шагов (месяцев)
    n_output = 24   # прогноз на следующие 6 временных шагов

    encoder_X, decoder_y = create_sequences(X_scaled, y_scaled, n_input, n_output)

    # 4. Разделение на обучающую и тестовую выборки (80% / 20%)
    split_idx = int(0.9 * len(encoder_X))
    X_train, X_test = encoder_X[:split_idx], encoder_X[split_idx:]
    y_train, y_test = decoder_y[:split_idx], decoder_y[split_idx:]

    # 5. Построение модели на основе TCN вместо LSTM
    latent_dim = 200  # число фильтров (выходное число признаков) в TCN

    # Энкодер: TCN слой обрабатывает входную последовательность и возвращает последнее представление.
    encoder_inputs = Input(shape=(n_input, n_features), name="encoder_inputs")
    encoder = TCN(nb_filters=latent_dim,
                kernel_size=4,
                dilations=[1, 2, 4, 8],
                dropout_rate=0.1,
                return_sequences=False,
                name="encoder_tcn")(encoder_inputs)
    # encoder имеет форму (batch_size, latent_dim)

    # Для декодера повторим выход энкодера n_output раз, чтобы сформировать входную последовательность фиксированной длины.
    decoder_inputs = RepeatVector(n_output, name="repeat_vector")(encoder)
    # decoder_inputs имеет форму (batch_size, n_output, latent_dim)

    # Декодер: TCN слой, возвращающий последовательность предсказаний.
    decoder = TCN(nb_filters=latent_dim,
                kernel_size=4,
                dilations=[1, 2, 4, 8],
                dropout_rate=0.1,
                return_sequences=True,
                name="decoder_tcn")(decoder_inputs)

    # Выходной слой, применяемый по времени для получения итогового прогноза Freight_Price (1 признак)
    decoder_outputs = TimeDistributed(Dense(1, activation='linear'), name="time_dist_dense")(decoder)

    # Собираем модель
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # 6. Обучение модели
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=64,
                        validation_split=0.1)

    # 7. Прогнозирование и визуализация результатов
    # Получаем прогноз для одного примера тестовой выборки
    pred_df = pd.DataFrame(columns=["Freight_Price"])
    # for i in range(len(X_test) - 1):
    #     # Получаем входной пример для тестовой выборки
    sample_input = X_test[22:23]  # shape: (1, n_input, n_features)
    print(f"Форма входного примера: {X_test.shape}")
    predicted_seq = model.predict(sample_input)

    # Приводим предсказания и истинные значения к исходному масштабу Freight_Price
    predicted_seq_inv = scaler_y.inverse_transform(predicted_seq[0])
    y_true_inv = scaler_y.inverse_transform(y_test[0])

    # Вычисление MAE и RMSE (вычисляем по всем временным шагам тестовой выборки)
    mae = mean_absolute_error(y_true_inv.flatten(), predicted_seq_inv.flatten())
    rmse = math.sqrt(mean_squared_error(y_true_inv.flatten(), predicted_seq_inv.flatten()))

    print("Метрики для тестовой выборки:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print("Прогноз Freight_Price:")
    pred_df = pd.concat([pred_df, pd.DataFrame(predicted_seq_inv, columns=["Freight_Price"])])
    pred_df.index = pd.date_range(start=df.index[split_idx + n_input + 22], periods=n_output, freq='M')
    return pred_df, df
    # print("\nИстинные значения Freight_Price:")
    # print(y_true_inv)

# plt.figure(figsize=(8, 4))
# plt.plot(pred_df, label='Прогноз', color='red', linestyle='--')
# plt.plot(df['Freight_Price'], label='Истинное значение', color='blue')
# plt.title("Прогноз фрахта (SEQ2SEQ)")
# plt.xlabel("Шаг времени")
# plt.ylabel("Цена фрахта")
# plt.legend()
# plt.show()
