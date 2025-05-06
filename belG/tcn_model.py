import numpy as np
import pandas as pd
import tensorflow as tf
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
import math
import matplotlib.pyplot as plt
from itertools import product
from keras.losses import MeanAbsoluteError, Loss, MeanSquaredError

# Параметры модели и путям к файлам
N_INPUT = 24
N_OUTPUT = 24
MODEL_PATH = 'belG/tcn_weights.h5'
SCALER_X_PATH = 'belG/scaler_X.pkl'
SCALER_Y_PATH = 'belG/scaler_y.pkl'


def check_device():
    print(tf.__version__)
    devices = tf.config.list_physical_devices('GPU')
    print("Using GPU" if devices else "Using CPU")


def load_data(path: str, oil_path) -> pd.DataFrame:
    # df_oil = pd.read_csv(oil_path, parse_dates=["Date"])
    # df_oil.drop(columns=["Open","High","Low","Vol.","Change %"], inplace=True)
    # df_oil.sort_values('Date', inplace=True)
    df = pd.read_csv(path, parse_dates=["Date"])
    # df = df.merge(df_oil, on='Date', how='left')
    df['Oil_Lag6']     = df['Oil_Lag2'].shift(4)
    df['Freight_Lag6'] = df['Freight_Lag2'].shift(4)
    df.dropna(inplace=True)
    # df = pd.get_dummies(df, columns=['has_crisis', 'has_war', 'has_sanctions', 'has_pandemic'], drop_first=False)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df


def preprocess(
    df: pd.DataFrame,
    time_cols: list,
    cat_cols: list,
    val_size: int,
    test_size: int,
    n_input: int
) -> tuple:
    """
    Делит DF на train/val/test с дополнительными n_input строками
    в начале каждого из val/test, чтобы create_sequences вернул хотя бы
    по одному окну. Фитит скейлеры только на train.
    """

    # 1) Назначаем список признаков и таргета
    target_col = 'Freight_Price'
    feat_cols  = [c for c in time_cols + cat_cols if c != target_col]

    df = df[feat_cols + [target_col]].dropna().sort_index()
    total = len(df)

    # 2) Индексы разбиения
    train_end  = total - val_size - test_size
    val_start  = train_end - n_input
    val_end    = total - test_size
    test_start = total - test_size - n_input

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[val_start:val_end]
    df_test  = df.iloc[test_start:]

    # 3) Фитим скейлеры только на train
    scaler_X = StandardScaler().fit(df_train[feat_cols])
    scaler_y = StandardScaler().fit(df_train[[target_col]])

    # 4) Трансформируем каждый датасет
    def _transform(subdf):
        X = scaler_X.transform(subdf[feat_cols])
        y = scaler_y.transform(subdf[[target_col]])
        return X, y

    X_train, y_train = _transform(df_train)
    X_val,   y_val   = _transform(df_val)
    X_test,  y_test  = _transform(df_test)

    return (X_train, X_val, X_test), (y_train, y_val, y_test), scaler_X, scaler_y


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    n_input: int,
    n_output: int
) -> tuple:
    enc, dec = [], []
    total = len(X) - n_input - n_output + 1
    for i in range(total):
        enc.append(X[i : i + n_input])
        dec.append(y[i + n_input : i + n_input + n_output])
    return np.array(enc), np.array(dec)

class DerivativeMatchingLoss(Loss):
    def __init__(self, alpha=1.0, name='derivative_matching_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        # 1) базовый MAE на всю последовательность
        loss_base = self.mae(y_true, y_pred)

        # 2) первая разность: форма (batch, T-1, 1)
        dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]

        # 3) MSE между этими разностями
        loss_dyn = self.mse(dy_true, dy_pred)

        return loss_base + self.alpha * loss_dyn
    
class VarianceRegularizerLoss(Loss):
    def __init__(self, beta=0.1, eps=1e-6, name='variance_regularizer_loss'):
        super().__init__(name=name)
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.beta = beta
        self.eps = eps

    def call(self, y_true, y_pred):
        # 1) стандартный MAE
        loss_base = self.mae(y_true, y_pred)

        # 2) считаем дисперсию предсказаний по временной оси
        # y_pred: (batch, T, 1) → var: (batch, 1)
        var = tf.math.reduce_variance(y_pred, axis=1)
        # убираем лишнюю размерность → (batch,)
        var = tf.squeeze(var, axis=-1)

        # 3) средний обратный дисперсия (штраф за низкую Var)
        reg = tf.reduce_mean(1.0 / (var + self.eps))

        return loss_base + self.beta * reg

def build_model(
    n_input: int,
    n_features: int,
    n_output: int,
    enc_filters: int = 512,
    enc_kernel_size: int = 2,
    enc_dilations_groups: list = [[1,2], [1,2,4]],
    enc_dropout: float = 0.0,
    dec_filters: int = 512,
    dec_kernel_size: int = 2,
    dec_dilations_groups: list = [[1,2], [1,2,4]],
    dec_dropout: float = 0.0,
    learning_rate: float = 1e-3,
    k_attention: int = 4
) -> tf.keras.Model:
    """
    Строит Seq2Seq-модель на основе TCN с несколькими блоками дилатаций и attention.
    """
    # Вход энкодера
    enc_inputs = tf.keras.layers.Input((n_input, n_features), name='encoder_inputs')
    x = enc_inputs
    # Последовательность TCN-блоков энкодера
    for idx, dilations in enumerate(enc_dilations_groups):
        x = TCN(
            nb_filters=enc_filters,
            kernel_size=enc_kernel_size,
            dilations=dilations,
            dropout_rate=enc_dropout,
            return_sequences=True,
            name=f'encoder_tcn_{idx}'
        )(x)
    enc_seq = x  # форма (batch, n_input, enc_filters)

    # Last encoder output for initialization
    enc_last = tf.keras.layers.Lambda(lambda z: z[:, -1, :], name='last_encoding')(enc_seq)
    dec_init = tf.keras.layers.RepeatVector(n_output, name='repeat_vector')(enc_last)

    # Attention: между всеми шагами декодера и энкодера
    recent_enc = enc_seq[:, -k_attention:, :]
    context = tf.keras.layers.Attention(name='attention')([dec_init, recent_enc])
    dec_input = tf.keras.layers.Concatenate(name='concat_attention')([dec_init, context])

    # Последовательность TCN-блоков декодера
    y = dec_input
    for idx, dilations in enumerate(dec_dilations_groups):
        y = TCN(
            nb_filters=dec_filters,
            kernel_size=dec_kernel_size,
            dilations=dilations,
            dropout_rate=dec_dropout,
            return_sequences=True,
            name=f'decoder_tcn_{idx}'
        )(y)

    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='linear'),
        name='time_dist_dense'
    )(y)

    model = tf.keras.Model(enc_inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.RootMeanSquaredError(name='rmse')
    ]
    model.compile(optimizer=optimizer, loss=VarianceRegularizerLoss(beta=1), metrics=metrics)
    return model


def save_artifacts(
    model: tf.keras.Model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler
) -> None:
    model.save_weights(MODEL_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print(f"Saved model weights to '{MODEL_PATH}', scalers to '{SCALER_X_PATH}', '{SCALER_Y_PATH}'")


def load_artifacts(
    n_input: int,
    n_features: int,
    n_output: int,
    enc_filters: int,
    enc_kernel_size: int,
    enc_dilations: list,
    enc_dropout: float,
    dec_filters: int,
    dec_kernel_size: int,
    dec_dilations: list,
    dec_dropout: float,
    learning_rate: float
) -> tuple:
    model = build_model(
        n_input, n_features, n_output,
        enc_filters, enc_kernel_size, enc_dilations, enc_dropout,
        dec_filters, dec_kernel_size, dec_dilations, dec_dropout,
        learning_rate
    )
    model.load_weights(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print("Loaded model and scalers from disk.")
    return model, scaler_X, scaler_y


def predict_sequence(
    sequence_array: np.ndarray,
    enc_filters: int,
    enc_kernel_size: int,
    enc_dilations: list,
    enc_dropout: float,
    dec_filters: int,
    dec_kernel_size: int,
    dec_dilations: list,
    dec_dropout: float,
    learning_rate: float
) -> np.ndarray:
    model, scaler_X, scaler_y = load_artifacts(
        N_INPUT, sequence_array.shape[1], N_OUTPUT,
        enc_filters, enc_kernel_size, enc_dilations, enc_dropout,
        dec_filters, dec_kernel_size, dec_dilations, dec_dropout,
        learning_rate
    )
    inp = sequence_array.reshape(1, N_INPUT, sequence_array.shape[1])
    pred_scaled = model.predict(inp)
    return scaler_y.inverse_transform(pred_scaled[0])



def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_input: int,
    n_features: int,
    n_output: int,
    param_grid: dict,
    validation_split: float = 0.1,
    save_path: str = 'best_params.json'
) -> dict:
    """
    Перебирает все комбинации из param_grid,
    обучает модель с validation_split на train,
    выбирает лучшую по минимальному val_loss (MSE) и сохраняет в best_params.json.
    """
    keys = list(param_grid.keys())
    best_loss = float('inf')
    best_params = {}

    for combo in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        print(f"Testing params: {params}")
        model = build_model(
            n_input, n_features, n_output,
            params['enc_filters'],    params['enc_kernel_size'],    params['enc_dilations'],    params['enc_dropout'],
            params['dec_filters'],    params['dec_kernel_size'],    params['dec_dilations'],    params['dec_dropout'],
            params['learning_rate'],
            k_attention=params.get('k_attention', 2)
        )
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=validation_split,
            verbose=0
        )
        val_loss = history.history['val_loss'][-1]
        print(f"  val_loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params.copy()

    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best params saved to '{save_path}': {best_params}, val_loss: {best_loss:.4f}")
    return best_params


def plot_history(history) -> None:
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def evaluate_and_plot(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: StandardScaler,
    start_date: pd.Timestamp
) -> None:
    pred = model.predict(X_test[:1])
    pred_inv = scaler_y.inverse_transform(pred[0])
    true_inv = scaler_y.inverse_transform(y_test[0])

    mae  = mean_absolute_error(true_inv.flatten(), pred_inv.flatten())
    rmse = math.sqrt(mean_squared_error(true_inv.flatten(), pred_inv.flatten()))
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    dates = pd.date_range(start=start_date, periods=pred_inv.shape[0], freq='M')
    df_plot = pd.DataFrame({
        'True': true_inv.flatten(),
        'Predicted': pred_inv.flatten()
    }, index=dates)
    # df_plot.plot(title='Freight Price: True vs Predicted')
    plt.plot(df_plot.index, df_plot['True'], label='True', color='blue')
    plt.plot(df_plot.index, df_plot['Predicted'], label='Predicted', color='orange')
    plt.show()


def main(train: bool = True, tune: bool = False):
    check_device()
    df = load_data('data/ML.csv', 'data\monthly_oil_cost_1988-2025.csv')
    time_cols = [
        'Freight_Price', 'Oil_Price',
        'Freight_Lag1','Freight_Lag2',
        'Oil_Lag1','Oil_Lag2',
        'Oil_Lag6','Freight_Lag6'
    ]
    cat_cols = [
        'has_crisis','has_war'
    ]

    # 1) Preprocess + split + scale
    (X_train, X_val, X_test), (y_train, y_val, y_test), scaler_X, scaler_y = \
        preprocess(df, time_cols, cat_cols,
                val_size=N_OUTPUT,
                test_size=N_OUTPUT,
                n_input=N_INPUT)

    enc_X_train, dec_y_train = create_sequences(X_train, y_train, N_INPUT, N_OUTPUT)
    enc_X_val,   dec_y_val   = create_sequences(X_val,   y_val,   N_INPUT, N_OUTPUT)
    enc_X_test,  dec_y_test  = create_sequences(X_test,  y_test,  N_INPUT, N_OUTPUT)

    import time
    start = time.time()
    if tune:
        param_grid = {
            'enc_filters':     [256],
            'enc_kernel_size': [4],
            'enc_dilations': [
                # две TCN-группы
                [[1, 2],       [1, 2, 4]],
                [[1, 2, 4],    [1, 2, 4, 8]],
                [[1, 2, 4, 8], [1, 2, 4, 8, 16]],

                # три TCN-группы
                [[1, 2],       [1, 2, 4],    [1, 2, 4, 8]],
                [[1, 2, 4],    [1, 2, 4, 8], [1, 2, 4, 8, 16]],
            ],
            'enc_dropout':     [0.0],

            'dec_filters':     [256],
            'dec_kernel_size': [4],
            'dec_dilations': [
                [[1, 2],       [1, 2, 4]],
                [[1, 2, 4],    [1, 2, 4, 8]],
                [[1, 2, 4, 8], [1, 2, 4, 8, 16]],
                [[1, 2],       [1, 2, 4],    [1, 2, 4, 8]],
                [[1, 2, 4],    [1, 2, 4, 8], [1, 2, 4, 8, 16]],
            ],
            'dec_dropout':     [0.0],

            'learning_rate':   [1e-3],
            'batch_size':      [64],
            'epochs':          [100],
            'k_attention':     [2, 4, 6],
        }
        best_params = tune_hyperparameters(
            enc_X_train, dec_y_train,
            N_INPUT, X_train.shape[1], N_OUTPUT,
            param_grid,
            validation_split=0.1,
            save_path='best_params.json'
        )
        end = time.time()
        print(f"Hyperparameter tuning took {end - start:.2f} seconds")

        # 3b) Train final model on train+val
        X_comb = np.concatenate([enc_X_train, enc_X_val], axis=0)
        y_comb = np.concatenate([dec_y_train, dec_y_val], axis=0)

        final_model = build_model(
            N_INPUT, X_train.shape[1], N_OUTPUT,
            best_params['enc_filters'],
            best_params['enc_kernel_size'],
            best_params['enc_dilations'],
            best_params['enc_dropout'],
            best_params['dec_filters'],
            best_params['dec_kernel_size'],
            best_params['dec_dilations'],
            best_params['dec_dropout'],
            best_params['learning_rate'],
            k_attention=best_params['k_attention']
        )

        print(final_model.summary())

        final_model.fit(
            X_comb, y_comb,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            verbose=1
        )
        # save after training
        save_artifacts(final_model, scaler_X, scaler_y)

    elif train:
        # 4) Train-only branch: load best_params.json, train on train+val, save
        with open('belG/best_params.json', 'r') as f:
            best_params = json.load(f)

        X_comb = np.concatenate([enc_X_train, enc_X_val], axis=0)
        y_comb = np.concatenate([dec_y_train, dec_y_val], axis=0)

        final_model = build_model(
            N_INPUT, X_train.shape[1], N_OUTPUT,
            best_params['enc_filters'],
            best_params['enc_kernel_size'],
            best_params['enc_dilations'],
            best_params['enc_dropout'],
            best_params['dec_filters'],
            best_params['dec_kernel_size'],
            best_params['dec_dilations'],
            best_params['dec_dropout'],
            best_params['learning_rate'],
            k_attention=best_params['k_attention']
        )
        final_model.fit(
            X_comb, y_comb,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            verbose=1
        )
        save_artifacts(final_model, scaler_X, scaler_y)

    else:
        # 5) Inference-only: load model weights
        with open('belG/best_params.json', 'r') as f:
            best_params = json.load(f)
        final_model = build_model(
            N_INPUT, X_train.shape[1], N_OUTPUT,
            best_params['enc_filters'],
            best_params['enc_kernel_size'],
            best_params['enc_dilations'],
            best_params['enc_dropout'],
            best_params['dec_filters'],
            best_params['dec_kernel_size'],
            best_params['dec_dilations'],
            best_params['dec_dropout'],
            best_params['learning_rate'],
            k_attention=best_params['k_attention']
        )
        final_model.load_weights(MODEL_PATH)

    # 6) Final evaluation on test set
    print("Test evaluation:")
    results = final_model.evaluate(enc_X_test, dec_y_test, verbose=0)
    evaluate_and_plot(
        final_model, enc_X_test, dec_y_test,
        scaler_y, df.index[-N_OUTPUT]
    )
    print(dict(zip(final_model.metrics_names, results)))


if __name__ == '__main__':
    main(train=True, tune=False)
