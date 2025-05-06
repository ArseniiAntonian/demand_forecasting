import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
import math
import matplotlib.pyplot as plt
from itertools import product
from keras.losses import MeanAbsoluteError, Loss

# Paths and constants
N_INPUT = 24
N_OUTPUT = 24
MODEL_WEIGHTS = 'gru_weights.h5'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
PARAMS_PATH = 'belG/best_params_gru.json'

# GRU-based Seq2Seq builder with state and sequence projection

def build_model_gru(

    n_input: int,
    n_features: int,
    n_output: int,
    enc_units: int = 256,
    enc_layers: int = 2,
    dec_units: int = 256,
    dec_layers: int = 2,
    learning_rate: float = 1e-3,
    k_attention: int = 4
) -> tf.keras.Model:

    # 1) Encoder
    enc_inputs = tf.keras.layers.Input((n_input, n_features), name='encoder_inputs')
    x = enc_inputs
    enc_states = []
    for i in range(enc_layers):
        gru_enc = tf.keras.layers.GRU(
            enc_units,
            return_sequences=True,
            return_state=True,
            name=f'encoder_gru_{i}')
        
        x, state = gru_enc(x)
        enc_states.append(state)
    enc_last = enc_states[-1]  # shape (batch, enc_units)


    # 2) Project encoder state -> decoder units
    if enc_units != dec_units:
        dec_init_state = tf.keras.layers.Dense(
            dec_units,
            activation='tanh',
            name='state_projection')(enc_last)
    else:
        dec_init_state = enc_last


    # 3) Project encoder sequence -> decoder units for attention
    if enc_units != dec_units:
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(dec_units, activation='linear'),
            name='seq_projection')(x)

    # 4) Prepare decoder input with attention
    dec_init = tf.keras.layers.RepeatVector(n_output, name='repeat_vector')(dec_init_state)
    recent_enc = x[:, -k_attention:, :]
    context = tf.keras.layers.Attention(name='attention')([dec_init, recent_enc])
    dec_input = tf.keras.layers.Concatenate(name='concat_attention')([dec_init, context])


    # 5) Decoder stack
    y = dec_input
    for i in range(dec_layers):
        if i == 0:
            gru_dec = tf.keras.layers.GRU(
                dec_units,
                return_sequences=True,
                return_state=True,
                name=f'decoder_gru_{i}'
            )
            y, _ = gru_dec(y, initial_state=dec_init_state)
        else:
            gru_dec = tf.keras.layers.GRU(
                dec_units,
                return_sequences=True,
                name=f'decoder_gru_{i}'
            )
            y = gru_dec(y)

    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='linear'),
        name='time_dist_dense')(y)
    
    model = tf.keras.Model(enc_inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=[MeanAbsoluteError(name='mae'), tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model

# Data loading and preprocessing
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    df['Oil_Lag6']     = df['Oil_Lag2'].shift(4)
    df['Freight_Lag6'] = df['Freight_Lag2'].shift(4)
    df.dropna(inplace=True)
    df = pd.get_dummies(
        df,
        columns=['has_crisis', 'has_war', 'has_sanctions', 'has_pandemic'],
        drop_first=False
    )
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
    target = 'Freight_Price'
    feats = [c for c in time_cols + cat_cols if c != target]
    df = df[feats + [target]].dropna().sort_index()
    total = len(df)
    train_end  = total - val_size - test_size
    val_start  = train_end - n_input
    val_end    = total - test_size
    test_start = total - test_size - n_input
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[val_start:val_end]
    test_df  = df.iloc[test_start:]
    scaler_X = StandardScaler().fit(train_df[feats])
    scaler_y = StandardScaler().fit(train_df[[target]])
    def _transform(subdf):
        X = scaler_X.transform(subdf[feats])
        y = scaler_y.transform(subdf[[target]])
        return X, y
    X_train, y_train = _transform(train_df)
    X_val,   y_val   = _transform(val_df)
    X_test,  y_test  = _transform(test_df)
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

# Save/load artifacts
def save_artifacts(model: tf.keras.Model, scaler_X, scaler_y) -> None:
    model.save_weights(MODEL_WEIGHTS)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print(f"Saved weights to {MODEL_WEIGHTS}, scalers to {SCALER_X_PATH}, {SCALER_Y_PATH}")


def load_artifacts(
    n_input, n_features, n_output,
    enc_units, enc_layers, dec_units, dec_layers,
    learning_rate, k_attention
) -> tuple:
    model = build_model_gru(
        n_input, n_features, n_output,
        enc_units, enc_layers,
        dec_units, dec_layers,
        learning_rate, k_attention
    )
    model.load_weights(MODEL_WEIGHTS)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print("Loaded artifacts.")
    return model, scaler_X, scaler_y

# Evaluate & plot updated to use sequences
def evaluate_and_plot(
    model: tf.keras.Model,
    enc_X_test: np.ndarray,
    dec_y_test: np.ndarray,
    scaler_y: StandardScaler,
    start_date: pd.Timestamp
) -> None:
    pred = model.predict(enc_X_test[:1])
    pred_inv = scaler_y.inverse_transform(pred[0])
    true_inv = scaler_y.inverse_transform(dec_y_test[0])
    mae = mean_absolute_error(true_inv.flatten(), pred_inv.flatten())
    rmse = math.sqrt(mean_squared_error(true_inv.flatten(), pred_inv.flatten()))
    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}")
    dates = pd.date_range(start=start_date, periods=N_OUTPUT, freq='M')
    plt.plot(dates, true_inv.flatten(), label='True')
    plt.plot(dates, pred_inv.flatten(), label='Predicted')
    plt.legend()
    plt.show()

# Hyperparameter tuning adjusted for GRU
def tune_hyperparameters(
    X_train, y_train, n_input, n_features, n_output,
    param_grid: dict, validation_split: float = 0.1, save_path: str = PARAMS_PATH
) -> dict:
    keys = list(param_grid.keys())
    best_loss = float('inf')
    best = {}
    for combo in product(*[param_grid[k] for k in keys]):
        p = dict(zip(keys, combo))
        print(f"Testing {p}")
        enc_train, dec_train = create_sequences(X_train, y_train, n_input, n_output)
        model = build_model_gru(
            n_input, n_features, n_output,
            p['enc_units'], p['enc_layers'],
            p['dec_units'], p['dec_layers'],
            p['learning_rate'], p['k_attention']
        )
        hist = model.fit(
            enc_train, dec_train,
            epochs=p['epochs'], batch_size=p['batch_size'],
            validation_split=validation_split, verbose=0
        )
        val_loss = hist.history['val_loss'][-1]
        print(f" val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best = p.copy()
    with open(save_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"Best params saved: {best}")
    return best

# Main entry
def main(train=True, tune=False):
    df = load_data('data/ML.csv')
    time_cols = ['Freight_Price','Oil_Price','Freight_Lag1','Freight_Lag2','Oil_Lag1','Oil_Lag2','Oil_Lag6','Freight_Lag6']
    cat_cols  = [c for c in df.columns if 'has_' in c]
    (X_train, X_val, X_test), (y_train, y_val, y_test), sx, sy = preprocess(df, time_cols, cat_cols, N_OUTPUT, N_OUTPUT, N_INPUT)
    enc_X_train, dec_y_train = create_sequences(X_train, y_train, N_INPUT, N_OUTPUT)
    enc_X_val,   dec_y_val   = create_sequences(X_val,   y_val,   N_INPUT, N_OUTPUT)
    enc_X_test,  dec_y_test  = create_sequences(X_test,  y_test,  N_INPUT, N_OUTPUT)

    if tune:
        grid = {
            'enc_units':[128,256], 'enc_layers':[1,2],
            'dec_units':[128,256], 'dec_layers':[1,2],
            'learning_rate':[1e-3,3e-3], 'k_attention':[2,4],
            'batch_size':[32,64], 'epochs':[20,50]
        }
        _ = tune_hyperparameters(X_train, y_train, N_INPUT, X_train.shape[1], N_OUTPUT, grid)

    if train:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        Xc = np.concatenate([X_train, X_val], axis=0)
        yc = np.concatenate([y_train, y_val], axis=0)
        enc_X_c, dec_y_c = create_sequences(Xc, yc, N_INPUT, N_OUTPUT)
        model = build_model_gru(
            N_INPUT, X_train.shape[1], N_OUTPUT,
            params['enc_units'], params['enc_layers'],
            params['dec_units'], params['dec_layers'],
            params['learning_rate'], params['k_attention']
        )
        model.fit(enc_X_c, dec_y_c, epochs=params['epochs'], batch_size=params['batch_size'], verbose=1)
        save_artifacts(model, sx, sy)
    else:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        model, _, _ = load_artifacts(
            N_INPUT, X_train.shape[1], N_OUTPUT,
            params['enc_units'], params['enc_layers'],
            params['dec_units'], params['dec_layers'],
            params['learning_rate'], params['k_attention']
        )

    print("Test evaluation:")
    test_loss = model.evaluate(enc_X_test, dec_y_test, verbose=1)
    print(dict(zip(model.metrics_names, test_loss)))
    evaluate_and_plot(model, enc_X_test, dec_y_test, sy, df.index[-N_OUTPUT])

if __name__ == '__main__':
    main(train=False, tune=False)
