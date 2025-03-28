import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR


def load_data(oil_data_path, freight_data_path):
    """Загружает и объединяет данные о нефти и фрахте."""
    oil_prices = pd.read_csv(oil_data_path, usecols=["Date", "Price"])
    freight_prices = pd.read_csv(freight_data_path, usecols=["Date", "Price"])
    
    # Преобразуем столбцы
    oil_prices["Price"] = oil_prices["Price"].astype(float)
    freight_prices["Price"] = freight_prices["Price"].replace(',', '', regex=True).astype(float)
    
    # Конвертируем даты и сортируем
    oil_prices["Date"] = pd.to_datetime(oil_prices["Date"], format="%m/%d/%Y")
    freight_prices["Date"] = pd.to_datetime(freight_prices["Date"], format="%m/%d/%Y")
    
    # Объединяем по дате
    df = pd.merge(freight_prices, oil_prices, on="Date", suffixes=("_freight", "_oil"))
    df.sort_values(by="Date", inplace=True)
    df.set_index("Date", inplace=True)
    df.index.freq = 'MS'  # Указываем месячную частоту
    
    return df


def check_stationarity(series):
    """Проверяет стационарность временного ряда с помощью ADF-теста."""
    adf_result = adfuller(series)
    return adf_result[1] < 0.05  # True, если ряд стационарен


def preprocess_data(df, seasonal_lag=12):
    """Проверяет стационарность и делает сезонное и обычное дифференцирование, если нужно."""
    for column in df.columns:
        if not check_stationarity(df[column]):
            df[column] = df[column].diff(seasonal_lag)
            df.dropna(inplace=True)  # Удаляем NaN
            
            if not check_stationarity(df[column]):
                df.loc[:, column] = df[column].diff()  # Убираем тренд
                df.dropna(inplace=True)  # Удаляем NaN снова

    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # На случай inf значений
    df.dropna(inplace=True)  # Финальная очистка
    return df


def plot_decomposition(series, period=12):
    """Выполняет декомпозицию временного ряда и строит графики."""
    result = seasonal_decompose(series, model="additive", period=period)
    result.plot()
    plt.show()


def plot_acf_pacf(series):
    """Строит графики автокорреляции и частичной автокорреляции."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, ax=axes[0])
    plot_pacf(series, ax=axes[1])
    plt.show()

def invert_differencing(original_series, differenced_series, interval=1):
    """Восстанавливает данные после дифференцирования."""
    inverted_series = differenced_series.copy()
    
    for i in range(interval, len(differenced_series)):
        inverted_series.iloc[i] = differenced_series.iloc[i] + original_series.iloc[i - interval]
    
    return inverted_series



class VARModel:
    def __init__(self, df, train_size=0.8, p=2):
        """Инициализация модели VAR."""
        self.df = df
        self.p = p
        split_index = int(len(df) * train_size)
        self.train_data, self.test_data = df.iloc[:split_index], df.iloc[split_index:]
        self.model = None
        self.results = None

    def fit(self):
        """Обучает модель VAR."""
        self.model = VAR(self.train_data)
        self.results = self.model.fit(self.p)
        print(self.results.summary())

    def forecast(self, steps=None):
        """Делает прогноз на тестовые данные."""
        if self.results is None:
            raise ValueError("Сначала обучите модель с помощью fit().")
        if steps is None:
            steps = len(self.test_data)
        forecast_values = self.results.forecast(self.train_data.values[-self.p:], steps=steps)
        return pd.DataFrame(forecast_values, index=self.test_data.index[:steps], columns=self.test_data.columns)

    def plot_results(self, forecast_steps=None):
        """Строит график прогнозов, восстановленных в исходные значения."""
        if self.results is None:
            raise ValueError("Сначала обучите модель с помощью fit().")

        if forecast_steps is None:
            forecast_steps = len(self.test_data)

        # Делаем прогноз
        forecast_df = self.forecast(forecast_steps)

        # Восстанавливаем предсказанные и тестовые значения
        forecast_df["Price_freight"] = invert_differencing(self.train_data["Price_freight"], forecast_df["Price_freight"], interval=12)
        test_data_restored = invert_differencing(self.train_data["Price_freight"], self.test_data["Price_freight"], interval=12)

        # Убираем NaN после восстановления
        forecast_df.dropna(inplace=True)
        test_data_restored.dropna(inplace=True)

        # График
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_data.index, self.train_data["Price_freight"], label="Train Data", color='blue')
        plt.plot(test_data_restored.index, test_data_restored, label="Test Data", color='green')
        plt.plot(forecast_df.index, forecast_df["Price_freight"], label="Forecast", linestyle="dashed", color='red')
        
        plt.xlabel("Date")
        plt.ylabel("Freight Price")
        plt.title("VAR Forecast vs. Actual Data (Restored Scale)")
        plt.legend()
        plt.show()




# --- Основной скрипт ---
if __name__ == "__main__":
    df = load_data("data/monthly_oil_cost_1988-2025.csv", "data/freight_cost.csv")
    df = preprocess_data(df)
    model = VARModel(df)
    model.fit()
    model.plot_results()
