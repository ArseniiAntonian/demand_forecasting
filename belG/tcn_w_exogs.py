from typing import List, Optional, Tuple
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
from metrics import TrendMetrics
import pywt

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
    # df = df.loc['2004-01-01': '2020-01-01']
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

class DerivativeMatchingLoss(tf.keras.losses.Loss):
    def __init__(self,
                 alpha=1.0,
                 w_financial=1.0,
                 w_pandemic=1.0,
                 w_geopolitical=1.0,
                 w_natural=1.0,
                 w_logistical=1.0,
                 name='derivative_matching_loss'):
        super().__init__(name=name)
        self.alpha          = alpha
        self.w_financial    = w_financial
        self.w_pandemic     = w_pandemic
        self.w_geopolitical = w_geopolitical
        self.w_natural      = w_natural
        self.w_logistical   = w_logistical
        self.eps            = tf.keras.backend.epsilon()

    def call(self, y_true, y_pred):
        # извлекаем значения и по порядку — все 5 масок
        y_vals        = y_true[..., 0:1]
        mask_fin      = y_true[..., 1:2]
        mask_pan      = y_true[..., 2:3]
        mask_geo      = y_true[..., 3:4]
        mask_nat      = y_true[..., 4:5]
        mask_log      = y_true[..., 5:6]

        # формируем общий вес в каждой точке времени
        weights = (
            1.0
            + (self.w_financial    - 1.0) * mask_fin
            + (self.w_pandemic     - 1.0) * mask_pan
            + (self.w_geopolitical - 1.0) * mask_geo
            + (self.w_natural      - 1.0) * mask_nat
            + (self.w_logistical   - 1.0) * mask_log
        )

        # базовый MAE с учётом весов
        abs_diff = tf.abs(y_vals - y_pred)
        loss_base = tf.reduce_sum(weights * abs_diff) / (tf.reduce_sum(weights) + self.eps)

        # динамическая часть (MSE по первым разностям), аналогично с усреднёнными весами
        dy_true   = y_vals[:,1:,:] - y_vals[:,:-1,:]
        dy_pred   = y_pred[:,1:,:] - y_pred[:,:-1,:]
        w1        = weights[:,1:,:]
        w0        = weights[:,:-1,:]
        weights_d = (w1 + w0) / 2.0
        loss_dyn  = tf.reduce_sum(weights_d * tf.square(dy_true - dy_pred)) \
                    / (tf.reduce_sum(weights_d) + self.eps)

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

class WaveletTransformLayer(tf.keras.layers.Layer):
    def __init__(self,
                 wavelet: str = 'db4',
                 level: int = 1,
                 thresholding: str = 'soft',
                 threshold_sigma: float = 2.0,
                 wavelet_neurons: int = 16,
                 init_scale: float = 1.0,
                 init_shift: float = 0.0,
                 scale_regularization: float = 0.0,
                 shift_regularization: float = 0.0,
                 mode='symmetric',
                 **kwargs):
        super().__init__(**kwargs)
        # сохраняем все параметры
        self.wavelet = wavelet
        self.level = level
        self.thresholding = thresholding
        self.threshold_sigma = threshold_sigma
        self.n_neurons = wavelet_neurons
        self.init_scale = init_scale
        self.init_shift = init_shift
        self.scale_reg = scale_regularization
        self.shift_reg = shift_regularization
        self.mode = mode

    def call(self, inputs):
        # inputs: shape (batch, timesteps, features)
        def denoise_series(x_np):
            # x_np: 1D numpy array (timesteps,)
            coeffs = pywt.wavedec(x_np, self.wavelet, mode=self.mode, level=self.level)
            # вычисляем порог по первой детальной компоненте
            sigma = self.threshold_sigma * np.std(coeffs[1])
            # пороговая обработка detail-коэффициентов
            if self.thresholding == 'soft':
                coeffs[1:] = [pywt.threshold(c, sigma, 'soft') for c in coeffs[1:]]
            else:
                coeffs[1:] = [pywt.threshold(c, sigma, 'hard') for c in coeffs[1:]]
            # реконструкция
            return pywt.waverec(coeffs, self.wavelet, mode=self.mode)

        def map_fn(sample):
            # sample: (timesteps, features)
            rec = []
            for i in range(sample.shape[1]):
                series = sample[:, i]
                den = tf.numpy_function(
                    func=denoise_series,
                    inp=[series],
                    Tout=tf.float32
                )
                den = tf.reshape(den, tf.shape(series))
                rec.append(den)
            # объединяем обратно в (timesteps, features)
            return tf.stack(rec, axis=-1)

        # применяем по каждому примеру в батче
        return tf.map_fn(map_fn, inputs)

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
    k_attention: int = 4,
    w_financial: float =8.0, 
    w_pandemic: float = 8.0, 
    w_geopolitical: float=8.0, 
    w_natural: float=8.0, 
    w_logistical: float=8.0,

    wavelet: str = 'db4',
    decomposition_level: int = 1,
    thresholding: str = 'soft',
    threshold_sigma: float = 2.0,
    wavelet_neurons: int = 16,
    init_scale: float = 1.0,
    init_shift: float = 0.0,
    scale_regularization: float = 0.0,
    shift_regularization: float = 0.0,
) -> tf.keras.Model:
    # 3.1 Энкодер
    enc_inputs = tf.keras.layers.Input((n_input, n_features), name='encoder_inputs')

    x = WaveletTransformLayer(wavelet=wavelet,
            level=decomposition_level,
            thresholding=thresholding,
            threshold_sigma=threshold_sigma,
            wavelet_neurons=wavelet_neurons,
            init_scale=init_scale,
            init_shift=init_shift,
            scale_regularization=scale_regularization,
            shift_regularization=shift_regularization,
            name='wavelet_denoise')(enc_inputs)

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
        loss=DerivativeMatchingLoss(alpha=0.5, w_financial=w_financial, w_pandemic=w_pandemic, w_geopolitical=w_geopolitical, w_natural=w_natural, w_logistical=w_logistical),
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
    n_trials: int = 10,
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
            params['learning_rate'], w_financial=params['w_financial'], w_geopolitical=params['w_geopolitical'],
            w_natural=params['w_natural'], w_logistical=params['w_logistical'],
            k_attention=params.get('k_attention', 2),
            wavelet=params.get('wavelet', 'db4'), decomposition_level= params.get('decomposition_level', 1),
            thresholding=params.get('thresholding', 'soft'), threshold_sigma=params.get('threshold_sigma', 2.0),
            wavelet_neurons=params.get('wavelet_neurons', 16), init_scale= params.get('init_scale', 1.0),
            init_shift=params.get('init_shift', 0.0), scale_regularization= params.get('scale_regularization', 0.0),
            shift_regularization=params.get('shift_regularization', 0.0)
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
def make_y_with_masks(y_vals: np.ndarray, exog: np.ndarray) -> np.ndarray:
    # exog.shape = (batch, T, n_exog)
    # cat_cols = [
    #   'has_crisis_y','crisis_intensity','crisis_shock',
    #   'crisis_type_Financial','crisis_type_Pandemic',
    #   'crisis_type_Geopolitical','crisis_type_Natural','crisis_type_Logistical'
    # ]
    # Индексы масок типов кризисов в этой последовательности:
    idxs = [3, 4, 5, 6, 7]
    masks = [exog[..., i:i+1] for i in idxs]
    # Собираем по каналу: [y_vals, mask_fin, mask_pan, mask_geo, mask_nat, mask_log]
    return np.concatenate([y_vals] + masks, axis=-1)

def filter_and_apply_crises(
    df_exog: pd.DataFrame,
    crisis_events: pd.DataFrame,
    include_types: Optional[List[str]]   = None,
    exclude_types: Optional[List[str]]   = None,
    include_events: Optional[List[str]]  = None,
    exclude_events: Optional[List[str]]  = None,
    include_period: Optional[Tuple[str,str]] = None,
    exclude_period: Optional[Tuple[str,str]] = None,
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    1) Фильтрует crisis_events по типам, именам и/или периодам.
    2) Обнуляет все кризисные флаги в df_exog.
    3) «Прорисовывает» отфильтрованные события в виде бинарных столбцов 
       и при наличии — Intensity/Shock.

    Аргументы:
        df_exog        – DataFrame с колонкой date_col и кризисными флагами;
        crisis_events  – DataFrame с ['Start','End','Name','Type', ...optional Intensity/Shock];
        include_types  – список Type, которые _хочем_ оставить;
        exclude_types  – список Type, которые _хочем_ убрать;
        include_events – список Name, которые оставить;
        exclude_events – список Name, которые убрать;
        include_period – (start,end) — брать только события, полностью в этом диапазоне;
        exclude_period – (start,end) — убрать все события, перекрывающие этот диапазон;
        date_col       – имя колонки с датой в df_exog.
    Возвращает:
        новый DataFrame с обновлёнными кризисными флагами.
    """
    # 1) Фильтрация списка событий
    ev = crisis_events.copy()
    ev["Start"] = pd.to_datetime(ev["Start"])
    ev["End"]   = pd.to_datetime(ev["End"])

    if include_types:
        ev = ev[ev["Type"].isin(include_types)]
    if exclude_types:
        ev = ev[~ev["Type"].isin(exclude_types)]
    if include_events:
        ev = ev[ev["Name"].isin(include_events)]
    if exclude_events:
        ev = ev[~ev["Name"].isin(exclude_events)]
    if include_period:
        s0,e0 = pd.to_datetime(include_period[0]), pd.to_datetime(include_period[1])
        ev = ev[(ev["Start"] >= s0) & (ev["End"] <= e0)]
    if exclude_period:
        s1,e1 = pd.to_datetime(exclude_period[0]), pd.to_datetime(exclude_period[1])
        ev = ev[~((ev["Start"] <= e1) & (ev["End"] >= s1))]

    # 2) Подготовка df_exog
    df = df_exog.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Соберём список всех кризисных столбцов (примеры ваших имён)
    crisis_flag_cols = [
        "has_crisis_y",
        "crisis_intensity",
        "crisis_shock",
        "crisis_type_Financial",
        "crisis_type_Pandemic",
        "crisis_type_Geopolitical",
        "crisis_type_Natural",
        "crisis_type_Logistical",
    ]
    # Обнуляем их
    for c in crisis_flag_cols:
        df[c] = 0

    # 3) Накладываем каждый отфильтрованный кризис
    for _, row in ev.iterrows():
        mask = (df[date_col] >= row["Start"]) & (df[date_col] <= row["End"])

        # всегда поднимаем общий флаг
        df.loc[mask, "has_crisis_y"] = 1

        # если есть Intensity/Shock в событиях, прокидываем
        if "Intensity" in row:
            df.loc[mask, "crisis_intensity"] = row["Intensity"]
        if "Shock" in row:
            df.loc[mask, "crisis_shock"] = row["Shock"]

        # прокидываем бинарный столбец по типу
        col_type = f"crisis_type_{row['Type']}"
        if col_type in df.columns:
            df.loc[mask, col_type] = 1

    return df

def main(train: bool = True, tune: bool = False, simulate: bool = False):
    """Main function to run the TCN model with exogenous variables.
    Args:
        train (bool): If True, trains the model.
        tune (bool): If True, performs hyperparameter tuning.
        simulate (bool): If True, runs crisis impact simulation.
    """
    check_device()
    df = load_data('data/ML_with_crisis.csv')

    time_cols = [
        'Freight_Price', 'Oil_Price',
        'Freight_Lag1','Freight_Lag2',
        'Oil_Lag1','Oil_Lag2',
        'Oil_Lag6','Freight_Lag6',
    ]
    cat_cols = ['has_crisis_y','crisis_intensity','crisis_shock','crisis_type_Financial','crisis_type_Pandemic','crisis_type_Geopolitical','crisis_type_Natural','crisis_type_Logistical']

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
            'w_financial' : [8.0, 1.0, 5.0], 
            'w_pandemic' : [8.0, 1.0, 5.0], 
            'w_geopolitical': [8.0, 1.0, 5.0], 
            'w_natural' : [8.0, 1.0, 5.0], 
            'w_logistical': [8.0, 1.0, 5.0],

            # Wavelet parameters
            'wavelet':                 ['db4', 'sym5', 'coif3'],
            'decomposition_level':     [1, 2, 3],
            'thresholding':            ['soft', 'hard'],
            'threshold_sigma':         [1.0, 2.0, 3.0],           # multiplier for σ in thresholding
            'wavelet_neurons':         [8, 16, 32, 64],          # number of ψ₍a,b₎ units
            'init_scale':              [0.5, 1.0, 2.0],          # initial aᵢ values
            'init_shift':              [0.0, 0.5, 1.0],          # initial bᵢ values
            'scale_regularization':    [0.0, 1e-4, 1e-3],        # L2 penalty on aᵢ
            'shift_regularization':    [0.0, 1e-4, 1e-3],        # L2 penalty on bᵢ
        }
        y_train_combined = make_y_with_masks(dec_y_train, dec_exog_train)
        best_params = tune_hyperparameters(
            enc_X_train, dec_exog_train, y_train_combined,
            N_INPUT, X_train.shape[1], N_OUTPUT, exog_train.shape[1],
            param_grid,
            n_trials=50,
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
            best_params['learning_rate'], k_attention=best_params['k_attention'],
            w_financial=best_params['w_financial'], w_geopolitical=best_params['w_geopolitical'],
            w_natural=best_params['w_natural'], w_logistical=best_params['w_logistical'],
            wavelet=best_params['wavelet'], decomposition_level=best_params['decomposition_level'],
            thresholding=best_params['thresholding'], threshold_sigma=best_params['threshold_sigma'],
            wavelet_neurons=best_params['wavelet_neurons'], init_scale=best_params['init_scale'],
            init_shift=best_params['init_shift'], scale_regularization=best_params['scale_regularization'],
            shift_regularization=best_params['shift_regularization']
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
            best_params['learning_rate'], k_attention=best_params['k_attention'],
            w_financial=best_params['w_financial'], w_geopolitical=best_params['w_geopolitical'],
            w_natural=best_params['w_natural'], w_logistical=best_params['w_logistical'],
            wavelet=best_params['wavelet'], decomposition_level=best_params['decomposition_level'],
            thresholding=best_params['thresholding'], threshold_sigma=best_params['threshold_sigma'],
            wavelet_neurons=best_params['wavelet_neurons'], init_scale=best_params['init_scale'],
            init_shift=best_params['init_shift'], scale_regularization=best_params['scale_regularization'],
            shift_regularization=best_params['shift_regularization']
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

    elif simulate:
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

        crisis_events = pd.DataFrame([
            {
                "Start":     "2008-09-01",
                "End":       "2009-03-31",
                "Name":      "Global Financial Crisis",
                "Type":      "Financial",
                "Intensity": 0.8,
                "Shock":     1.2
            },
            {
                "Start":     "2020-03-01",
                "End":       "2021-06-30",
                "Name":      "COVID-19 Pandemic",
                "Type":      "Pandemic",
                "Intensity": 1.0,
                "Shock":     0.9
            },
            {
                "Start":     "2014-03-01",
                "End":       "2015-12-31",
                "Name":      "Ukraine Sanctions Shock",
                "Type":      "Geopolitical",
                "Intensity": 0.6,
                "Shock":     0.7
            },
            {
                "Start":     "2011-03-11",
                "End":       "2011-05-31",
                "Name":      "Tohoku Earthquake & Tsunami",
                "Type":      "Natural",
                "Intensity": 0.5,
                "Shock":     0.4
            },
            {
                "Start":     "2022-06-01",
                "End":       "2022-08-31",
                "Name":      "Suez Canal Blockage",
                "Type":      "Logistical",
                "Intensity": 0.4,
                "Shock":     0.3
            }
        ])

        df_exog_filt = filter_and_apply_crises(
            dec_exog_test,
            crisis_events,
            exclude_types=["Geopolitical", "Logistical", ""],         # отключить геополитические
            exclude_period=("2020-03-01","2020-12-31"),  # отключить события 2020 года
        )

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
                best_params['learning_rate'], k_attention=best_params['k_attention'],
                w_financial=best_params['w_financial'], w_geopolitical=best_params['w_geopolitical'],
                w_natural=best_params['w_natural'], w_logistical=best_params['w_logistical'],
                wavelet=best_params['wavelet'], decomposition_level=best_params['decomposition_level'],
                thresholding=best_params['thresholding'], threshold_sigma=best_params['threshold_sigma'],
                wavelet_neurons=best_params['wavelet_neurons'], init_scale=best_params['init_scale'],
                init_shift=best_params['init_shift'], scale_regularization=best_params['scale_regularization'],
                shift_regularization=best_params['shift_regularization']
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

    crisis_types = {
    'crisis_type_Financial': 'red',
    'crisis_type_Pandemic': 'blue',
    'crisis_type_Geopolitical': 'green',
    'crisis_type_Natural': 'orange',
    'crisis_type_Logistical': 'purple'
    }

    def plot_crisis_bands(df, crisis_column, color):
        in_crisis = False
        is_shock = False
        intensity = 0
        start_date = None

        for idx, row in df.iterrows():
            if row[crisis_column] == 1 and not in_crisis:
                in_crisis = True
                intensity = row['crisis_intensity']
                if row['crisis_shock'] == 1:
                    is_shock = True
                start_date = idx
            elif row[crisis_column] == 0 and in_crisis:
                if is_shock:
                    plt.axvspan(start_date, idx, color=color, alpha=intensity, label=crisis_column if start_date == idx else "", hatch='/')
                else:
                    plt.axvspan(start_date, idx, color=color, alpha=intensity, label=crisis_column if start_date == idx else "")
                in_crisis = False
                is_shock = False


        if in_crisis:
            plt.axvspan(start_date, df.index[-1], color=color, alpha=0.3)



    plt.figure()
    plt.plot(df.index, df['Freight_Price'], label='True')
    plt.plot(df_plot.index, df_plot['Predicted'], label='Predicted')
    for crisis, color in crisis_types.items():
        plot_crisis_bands(df, crisis, color)
    plt.title('Freight Price: True vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


    df_plot.rename(columns={'True': 'Freight_Price', 'Predicted': 'yhat_exp'}, inplace=True)

    mae_val  = mean_absolute_error(true_inv.flatten(), pred_inv.flatten())
    rmse_val = math.sqrt(mean_squared_error(true_inv.flatten(), pred_inv.flatten()))
    metrics = TrendMetrics(df_plot)
    print(metrics.summary())
    print(f"MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}")


if __name__ == '__main__':
    main(train=True, tune=True, simulate=False)
