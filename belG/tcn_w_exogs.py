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
import optuna
from keras.optimizers import Adam
import time

# Параметры модели и путям к файлам
N_INPUT = 12
N_OUTPUT = 60
MODEL_PATH = 'belG/weights_w_exogs_5y/tcn_weights.h5'
SCALER_X_PATH = 'belG/weights_w_exogs_5y/scaler_X.pkl'
SCALER_Y_PATH = 'belG/weights_w_exogs_5y/scaler_y.pkl'


def check_device():
    print(tf.__version__)
    devices = tf.config.list_physical_devices('GPU')
    print("Using GPU" if devices else "Using CPU")


def load_data(path: str) -> pd.DataFrame:
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
    df = df.loc['2004-01-01': '2020-01-01']
    return df


def preprocess(
    df: pd.DataFrame,
    time_cols: list,
    cat_cols: list,
    val_size: int,
    test_size: int,
    n_input: int
) -> tuple:
    target_col = 'Freight_Price'
    # только временные признаки идут в энкодер
    feat_cols = [c for c in time_cols if c != target_col]
    df = df[feat_cols + cat_cols + [target_col]].dropna().sort_index()

    total = len(df)
    train_end  = total - val_size - test_size
    val_start  = train_end - n_input
    val_end    = total - test_size
    test_start = total - test_size - n_input

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[val_start:val_end]
    df_test  = df.iloc[test_start:]

    # скейлеры по только временным фичам и таргету
    scaler_X = StandardScaler().fit(df_train[feat_cols])
    scaler_y = StandardScaler().fit(df_train[[target_col]])

    def _transform(subdf):
        X = scaler_X.transform(subdf[feat_cols])
        y = scaler_y.transform(subdf[[target_col]])
        return X, y

    X_train, y_train = _transform(df_train)
    X_val,   y_val   = _transform(df_val)
    X_test,  y_test  = _transform(df_test)

    # и отдельно не­скейленные экзогены
    exog_train = df_train[cat_cols].values.astype(np.float32)
    exog_val   = df_val[cat_cols].values.astype(np.float32)
    exog_test  = df_test[cat_cols].values.astype(np.float32)

    return (X_train, X_val, X_test), \
           (y_train, y_val, y_test), \
           (exog_train, exog_val, exog_test), \
           scaler_X, scaler_y


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    exog: np.ndarray,
    n_input: int,
    n_output: int
) -> tuple:
    enc, dec, ex = [], [], []
    total = len(X) - n_input - n_output + 1
    for i in range(total):
        enc.append( X[i : i + n_input] )
        dec.append( y[i + n_input : i + n_input + n_output] )
        ex.append( exog[i + n_input : i + n_input + n_output] )
    return np.array(enc), np.array(dec), np.array(ex)

class DerivativeMatchingLoss(Loss):
    def __init__(self,
                 alpha=1.0,
                 crisis_weight=5.0,
                 war_weight=3.0,
                 name='derivative_matching_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.crisis_weight = crisis_weight
        self.war_weight = war_weight
        self.eps = tf.keras.backend.epsilon()

    def call(self, y_true, y_pred):
        # y_true: (batch, T, 3) -> [values, has_crisis, has_war]
        y_vals     = y_true[..., 0:1]
        crisis_mask= y_true[..., 1:2]
        war_mask   = y_true[..., 2:3]
        # составляем веса (аддитивно)
        weights = (1.0
                   + (self.crisis_weight - 1.0) * crisis_mask
                   + (self.war_weight    - 1.0) * war_mask)
        # базовый MAE с учётом весов
        abs_diff = tf.abs(y_vals - y_pred)
        loss_base = tf.reduce_sum(weights * abs_diff) / (tf.reduce_sum(weights) + self.eps)
        # динамическая часть (MSE по разностям)
        dy_true = y_vals[:, 1:, :] - y_vals[:, :-1, :]
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        sq_diff = tf.square(dy_true - dy_pred)
        w1 = weights[:, 1:, :]
        w0 = weights[:, :-1, :]
        weights_dyn = (w1 + w0) / 2.0
        loss_dyn = tf.reduce_sum(weights_dyn * sq_diff) / (tf.reduce_sum(weights_dyn) + self.eps)
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
    n_exog: int,
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
    # 3.1 Энкодер
    enc_inputs = tf.keras.layers.Input((n_input, n_features), name='encoder_inputs')
    x = enc_inputs
    for idx, dil in enumerate(enc_dilations_groups):
        x = TCN(nb_filters=enc_filters, kernel_size=enc_kernel_size,
                dilations=dil, dropout_rate=enc_dropout,
                return_sequences=True, name=f'enc_tcn_{idx}')(x)
    enc_seq  = x
    enc_last = tf.keras.layers.Lambda(lambda z: z[:, -1, :], name='last_encoding')(enc_seq)
    dec_init = tf.keras.layers.RepeatVector(n_output, name='repeat_vector')(enc_last)

    # 3.2 Механизм внимания на последние k_attention шагов
    recent_enc = enc_seq[:, -k_attention:, :]
    context    = tf.keras.layers.Attention(name='attention')([dec_init, recent_enc])

    # 3.3 Input для будущих экзогенов
    exog_inputs = tf.keras.layers.Input((n_output, n_exog), name='exog_inputs')

    # 3.4 Конкатенация декодерной инициализации, внимания и экзогенов
    dec_input = tf.keras.layers.Concatenate(name='concat_all')(
        [dec_init, context, exog_inputs]
    )

    # 3.5 Декодер (TCN-блоки)
    y = dec_input
    for idx, dil in enumerate(dec_dilations_groups):
        y = TCN(nb_filters=dec_filters, kernel_size=dec_kernel_size,
                dilations=dil, dropout_rate=dec_dropout,
                return_sequences=True, name=f'dec_tcn_{idx}')(y)

    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='linear'),
        name='time_dist_dense'
    )(y)

    model = tf.keras.Model([enc_inputs, exog_inputs], outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=DerivativeMatchingLoss(alpha=0.5, crisis_weight=10.0, war_weight=10.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                 tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
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
    n_exog: int,
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
        n_input, n_features, n_output, n_exog,
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
    enc_X_train: np.ndarray,
    dec_exog_train: np.ndarray,
    y_train: np.ndarray,
    n_input: int,
    n_features: int,
    n_output: int,
    n_exogs: int,
    param_grid: dict,
    n_trials: int = 100,
    validation_split: float = 0.2,
    save_path: str = 'best_params_optuna.json'
) -> dict:
    """
    Оптимизация гиперпараметров с помощью Optuna.
    Использует param_grid для определения диапазонов (categorical) и проводит n_trials запусков,
    выбирая параметры по минимальному val_loss (MSE) на валидационной выборке.
    Сохраняет лучшие параметры в save_path.
    """
    
    def objective(trial):
        # Формируем словарь параметров из param_grid
        params = {k: trial.suggest_categorical(k, param_grid[k]) for k in param_grid}

        # Строим модель с указанными гиперпараметрами
        model = build_model(
            n_input, n_features, n_output, n_exogs,
            params['enc_filters'], params['enc_kernel_size'], params['enc_dilations'], params['enc_dropout'],
            params['dec_filters'], params['dec_kernel_size'], params['dec_dilations'], params['dec_dropout'],
            params['learning_rate'],
            k_attention=params.get('k_attention', 2)
        )

        # Обучаем модель и возвращаем финальный val_loss
        history = model.fit(
            [enc_X_train, dec_exog_train], y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=validation_split,
            verbose=0
        )
        val_loss = history.history['val_loss'][-1]
        return val_loss

    # Создаем исследование Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Сохраняем лучшие параметры
    best_params = study.best_params
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"Best params saved to '{save_path}': {best_params}, val_loss: {study.best_value:.4f}")
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


# Помощник для склейки таргетов и масок
def make_y_with_masks(y_vals, exog):
    # y_vals: (batch, T, 1), exog: (batch, T, 2) -> [has_crisis, has_war]
    crisis = exog[..., 0:1]
    war    = exog[..., 1:2]
    return np.concatenate([y_vals, crisis, war], axis=-1)  # (batch, T, 3)

# Ваша main-функция с сохранением tune/train/inference

def main(train: bool = True, tune: bool = False):
    check_device()
    df = load_data('data/ML.csv')

    time_cols = [
        'Freight_Price', 'Oil_Price',
        'Freight_Lag1','Freight_Lag2',
        'Oil_Lag1','Oil_Lag2',
        'Oil_Lag6','Freight_Lag6',
    ]
    cat_cols = ['has_crisis','has_war']

    # 1) Preprocess + split + scale + exog
    (X_train, X_val, X_test), (y_train, y_val, y_test), \
        (exog_train, exog_val, exog_test), scaler_X, scaler_y = preprocess(
            df, time_cols, cat_cols,
            val_size=N_OUTPUT, test_size=N_OUTPUT, n_input=N_INPUT
        )

    # 2) Создание Seq2Seq-последовательностей
    enc_X_train, dec_y_train, dec_exog_train = create_sequences(
        X_train, y_train, exog_train, N_INPUT, N_OUTPUT
    )
    enc_X_val,   dec_y_val,   dec_exog_val   = create_sequences(
        X_val,   y_val,   exog_val,   N_INPUT, N_OUTPUT
    )
    enc_X_test,  dec_y_test,  dec_exog_test  = create_sequences(
        X_test,  y_test,  exog_test,  N_INPUT, N_OUTPUT
    )

    start = time.time()

    if tune:
        # --- Hyperparameter tuning ---
        param_grid = {
            'enc_filters':     [64, 128, 256],
            'enc_kernel_size': [2, 4, 8],
            'enc_dilations': [
                [[1,2], [1,2,4]],
                [[1,2,4], [1,2,4,8]],
                [[1,2,4,8], [1,2,4,8,16]],
                [[1, 2], [1, 2, 4], [1, 2, 4, 8]],
                [[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16]],
            ],
            'enc_dropout':     [0.0],
            'dec_filters':     [64, 128, 256],
            'dec_kernel_size': [2, 4, 8],
            'dec_dilations': [
                [[1,2], [1,2,4]],
                [[1,2,4], [1,2,4,8]],
                [[1,2,4,8], [1,2,4,8,16]],
                [[1, 2], [1, 2, 4], [1, 2, 4, 8]],
                [[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16]],
            ],
            'dec_dropout':     [0.0],
            'learning_rate':   [1e-3],
            'batch_size':      [32, 64],
            'epochs':          [100],
            'k_attention':     [3,6,8],
        }
        y_train_combined = make_y_with_masks(dec_y_train, dec_exog_train)
        best_params = tune_hyperparameters(
            enc_X_train, dec_exog_train, y_train_combined,
            N_INPUT, X_train.shape[1], N_OUTPUT, exog_train.shape[1],
            param_grid,
            n_trials=100,
            validation_split=0.2,
            save_path='belG/best_params_exogs_5y.json'
        )
        print(f"Hyperparameter tuning took {time.time() - start:.2f} seconds")

        # Train final on train+val
        X_comb    = np.concatenate([enc_X_train,  enc_X_val],  axis=0)
        exog_comb = np.concatenate([dec_exog_train, dec_exog_val], axis=0)
        y_vals_comb = np.concatenate([dec_y_train, dec_y_val], axis=0)
        y_combined = make_y_with_masks(y_vals_comb, exog_comb)

        final_model = build_model(
            N_INPUT, X_train.shape[1], N_OUTPUT, exog_train.shape[1],
            best_params['enc_filters'], best_params['enc_kernel_size'],
            best_params['enc_dilations'], best_params['enc_dropout'],
            best_params['dec_filters'], best_params['dec_kernel_size'],
            best_params['dec_dilations'], best_params['dec_dropout'],
            best_params['learning_rate'], k_attention=best_params['k_attention']
        )

        final_model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss=DerivativeMatchingLoss(alpha=1.0),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.RootMeanSquaredError(name='rmse')
                ]
        )
        final_model.fit(
            [X_comb, exog_comb], y_combined,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            verbose=1
        )
        save_artifacts(final_model, scaler_X, scaler_y)

    elif train:
        # --- Train-only: load best_params, train on train+val ---
        with open('belG/best_params_exogs_5y.json', 'r') as f:
            best_params = json.load(f)

        X_comb    = np.concatenate([enc_X_train,  enc_X_val],  axis=0)
        exog_comb = np.concatenate([dec_exog_train, dec_exog_val], axis=0)
        y_vals_comb = np.concatenate([dec_y_train, dec_y_val], axis=0)
        y_combined = make_y_with_masks(y_vals_comb, exog_comb)

        final_model = build_model(
            N_INPUT, X_train.shape[1], N_OUTPUT, exog_train.shape[1],
            best_params['enc_filters'], best_params['enc_kernel_size'],
            best_params['enc_dilations'], best_params['enc_dropout'],
            best_params['dec_filters'], best_params['dec_kernel_size'],
            best_params['dec_dilations'], best_params['dec_dropout'],
            best_params['learning_rate'], k_attention=best_params['k_attention']
        )

        final_model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss=DerivativeMatchingLoss(alpha=1.0),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.RootMeanSquaredError(name='rmse')
                ]
        )
        final_model.fit(
            [X_comb, exog_comb], y_combined,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            verbose=1
        )
        save_artifacts(final_model, scaler_X, scaler_y)

    else:
        # --- Inference-only: load model and weights ---
        with open('belG/best_params_exogs_5y.json', 'r') as f:
            best_params = json.load(f)
        final_model = build_model(
            N_INPUT, X_train.shape[1], N_OUTPUT, exog_train.shape[1],
            best_params['enc_filters'], best_params['enc_kernel_size'],
            best_params['enc_dilations'], best_params['enc_dropout'],
            best_params['dec_filters'], best_params['dec_kernel_size'],
            best_params['dec_dilations'], best_params['dec_dropout'],
            best_params['learning_rate'], k_attention=best_params['k_attention']
        )
        final_model.load_weights(MODEL_PATH)

    # --- Final evaluation on test set ---
    # Подготавливаем y_test с масками
    y_test_vals = dec_y_test
    y_test_combined = make_y_with_masks(y_test_vals, dec_exog_test)

    print("Test evaluation:")
    results = final_model.evaluate(
        [enc_X_test, dec_exog_test],
        y_test_combined,
        verbose=0
    )
    print(dict(zip(final_model.metrics_names, results)))

    # --- Plotting ---
    pred_scaled = final_model.predict([enc_X_test[:1], dec_exog_test[:1]])[0]
    pred_inv    = scaler_y.inverse_transform(pred_scaled)
    true_inv    = scaler_y.inverse_transform(dec_y_test[0])

    dates = pd.date_range(start=df.index[-N_OUTPUT], periods=N_OUTPUT, freq='M')
    df_plot = pd.DataFrame({
        'True':      true_inv.flatten(),
        'Predicted': pred_inv.flatten()
    }, index=dates)

    plt.figure()
    plt.plot(df.index, df['Freight_Price'], label='True')
    plt.plot(df_plot.index, df_plot['Predicted'], label='Predicted')
    plt.title('Freight Price: True vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    mae_val  = mean_absolute_error(true_inv.flatten(), pred_inv.flatten())
    rmse_val = math.sqrt(mean_squared_error(true_inv.flatten(), pred_inv.flatten()))
    print(f"MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")


if __name__ == '__main__':
    main(train=True, tune=True)
