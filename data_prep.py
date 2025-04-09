# Повторный импорт после сброса состояния
import pandas as pd
import numpy as np


# Загрузка файлов
freight_df = pd.read_csv('data/cleanFreight.csv')
oil_df = pd.read_csv('data/monthly_oil_cost_1988-2025.csv')

# Обработка даты
freight_df['Date'] = pd.to_datetime(freight_df['Date'], format='%m/%d/%Y')
oil_df['Date'] = pd.to_datetime(oil_df['Date'], format='%m/%d/%Y')

# Переименование и объединение
freight_df = freight_df.rename(columns={'Price': 'Freight_Price'})[['Date', 'Freight_Price']]
oil_df = oil_df.rename(columns={'Price': 'Oil_Price'})[['Date', 'Oil_Price']]
merged_df = pd.merge(freight_df, oil_df, on='Date', how='inner').sort_values(by='Date').reset_index(drop=True)
merged_df['Freight_Price'] = merged_df['Freight_Price'].astype(str).str.replace(',', '').astype(float)
merged_df['Oil_Price'] = merged_df['Oil_Price'].astype(float)

# Функция событий
def assign_event_labels(date):
    labels = {
        "crisis": None,
        "war": None,
        "sanctions": None,
        "pandemic": None
    }
    if pd.Timestamp('2000-03-01') <= date <= pd.Timestamp('2002-10-01'):
        labels["crisis"] = "Dot-com crash"
    elif pd.Timestamp('2008-09-01') <= date <= pd.Timestamp('2009-06-01'):
        labels["crisis"] = "Global financial crisis"
    elif pd.Timestamp('2014-10-01') <= date <= pd.Timestamp('2016-02-01'):
        labels["crisis"] = "Oil price collapse"
    elif pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2020-12-01'):
        labels["crisis"] = "COVID-19 oil crash"
    elif pd.Timestamp('2022-01-01') <= date <= pd.Timestamp('2023-12-01'):
        labels["crisis"] = "Inflation and rate hikes"
    if pd.Timestamp('2003-03-01') <= date <= pd.Timestamp('2003-12-01'):
        labels["war"] = "Iraq War"
    elif pd.Timestamp('2011-01-01') <= date <= pd.Timestamp('2012-12-01'):
        labels["war"] = "Arab Spring"
    elif pd.Timestamp('2014-03-01') <= date <= pd.Timestamp('2014-12-01'):
        labels["war"] = "Crimea crisis"
    elif pd.Timestamp('2022-02-01') <= date:
        labels["war"] = "Ukraine War"
    if pd.Timestamp('2012-01-01') <= date <= pd.Timestamp('2015-07-01'):
        labels["sanctions"] = "Iran sanctions"
    if pd.Timestamp('2014-03-01') <= date:
        labels["sanctions"] = "Russia sanctions"
    if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2022-01-01'):
        labels["pandemic"] = "COVID-19"
    return pd.Series(labels)

merged_df[['crisis', 'war', 'sanctions', 'pandemic']] = merged_df['Date'].apply(assign_event_labels)

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(merged_df[['crisis', 'war', 'sanctions', 'pandemic']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['crisis', 'war', 'sanctions', 'pandemic']))
ml_df = pd.concat([merged_df.drop(columns=['crisis', 'war', 'sanctions', 'pandemic']), encoded_df], axis=1)

# Временные признаки и технические индикаторы
ml_df['month'] = ml_df['Date'].dt.month
ml_df['year'] = ml_df['Date'].dt.year
ml_df['quarter'] = ml_df['Date'].dt.quarter
ml_df['is_year_start'] = ml_df['Date'].dt.is_year_start.astype(int)
ml_df['is_year_end'] = ml_df['Date'].dt.is_year_end.astype(int)
ml_df['sin_month'] = np.sin(2 * np.pi * ml_df['month'] / 12)
ml_df['cos_month'] = np.cos(2 * np.pi * ml_df['month'] / 12)
ml_df['Freight_Lag1'] = ml_df['Freight_Price'].shift(1)
ml_df['Freight_Lag2'] = ml_df['Freight_Price'].shift(2)
ml_df['Oil_Lag1'] = ml_df['Oil_Price'].shift(1)
ml_df['Oil_Lag2'] = ml_df['Oil_Price'].shift(2)
ml_df['Freight_SMA_3'] = ml_df['Freight_Price'].rolling(window=3).mean()
ml_df['Oil_SMA_3'] = ml_df['Oil_Price'].rolling(window=3).mean()
ml_df['has_crisis'] = (ml_df.filter(like="crisis_").drop(columns="crisis_None").sum(axis=1) > 0).astype(int)
ml_df['has_war'] = (ml_df.filter(like="war_").drop(columns="war_None").sum(axis=1) > 0).astype(int)
ml_df['has_sanctions'] = (ml_df.filter(like="sanctions_").drop(columns="sanctions_None").sum(axis=1) > 0).astype(int)
ml_df['has_pandemic'] = (ml_df.filter(like="pandemic_").drop(columns="pandemic_None").sum(axis=1) > 0).astype(int)

# Удаление NaN и генерация таргетов
ml_df = ml_df.dropna().reset_index(drop=True)
forecast_horizon = 3
for i in range(1, forecast_horizon + 1):
    ml_df[f'target_t+{i}'] = ml_df['Freight_Price'].shift(-i)
ml_df = ml_df.dropna().reset_index(drop=True)

# Разделение на X и y
target_cols = [f'target_t+{i}' for i in range(1, forecast_horizon + 1)]
X = ml_df.drop(columns=['Date'] + target_cols)
y = ml_df[target_cols]

# Train-test split
split_index = int(len(ml_df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# === Создание окон с переменной длиной ===
def create_tcn_sequences_variable_window(X_df, y_df, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(X_df)):
        X_window = X_df.iloc[i - window_size:i].values
        y_target = y_df.iloc[i].values
        X_seq.append(X_window)
        y_seq.append(y_target)
    return np.array(X_seq), np.array(y_seq)

# Размер окна можно менять здесь

# Применение
def get_data(custom_window_size):
    X_train_seq, y_train_seq = create_tcn_sequences_variable_window(X_train, y_train, custom_window_size)
    X_test_seq, y_test_seq = create_tcn_sequences_variable_window(X_test, y_test, custom_window_size)
    feature_names = X.columns.tolist()
    freight_idx = feature_names.index('Freight_Price')
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, freight_idx
