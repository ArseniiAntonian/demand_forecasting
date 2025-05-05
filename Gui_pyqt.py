import sys
import os
import importlib.util
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QListWidget,
    QMessageBox, QDateEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd

MODELS = {
    "Prophet" : "Ignat_prophet/Main.py",
    "Seq2seq" : "belG/lgbt.py",
    "LightGBM" : "arsen/gb_forecast.py"
}
EVENT_TYPES = ["Кризис", "Война"]
OIL_FORECAST_TYPES = ["walletinvestor", "EFA forecast"]

class ForecastWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("График прогноза фрахта")
        self.events = []
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        main_v = QVBoxLayout(central)

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        main_v.addWidget(self.canvas)
        main_v.addSpacing(40)
        main_v.addSpacing(20)

        controls = QHBoxLayout()

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(MODELS.keys())
        controls.addWidget(QLabel("Модель:"))
        controls.addWidget(self.cmb_model)

        self.cmb_oil_source = QComboBox()
        self.cmb_oil_source.addItems(OIL_FORECAST_TYPES)
        controls.addWidget(QLabel("Источник прогноза нефти:"))
        controls.addWidget(self.cmb_oil_source)

        self.txt_start = QDateEdit()
        self.txt_start.setDisplayFormat("Начало (yyyy-MM)")
        self.txt_start.setCalendarPopup(True)
        self.txt_start.setMinimumDate(QDate(1900, 1, 1))
        self.txt_start.setMaximumDate(QDate(2100, 12, 1))

        self.txt_end = QDateEdit()
        self.txt_end.setDisplayFormat("Конец (yyyy-MM)")
        self.txt_end.setCalendarPopup(True)
        self.txt_end.setMinimumDate(QDate(1900, 1, 1))
        self.txt_end.setMaximumDate(QDate(2100, 12, 1))

        controls.addWidget(self.txt_start)
        controls.addWidget(self.txt_end)

        self.cmb_type = QComboBox(); self.cmb_type.addItems(EVENT_TYPES)
        btn_add = QPushButton("Добавить событие")
        btn_add.clicked.connect(self.add_event)
        controls.addWidget(self.cmb_type)
        controls.addWidget(btn_add)

        self.lst_events = QListWidget()
        self.lst_events.setMaximumHeight(60)
        controls.addWidget(self.lst_events)

        btn_run = QPushButton("Запустить прогноз")
        btn_run.clicked.connect(self.run_model)
        controls.addWidget(btn_run)

        main_v.addLayout(controls)
        self.setCentralWidget(central)

    def add_event(self):
        start = self.txt_start.text().strip()
        end   = self.txt_end.text().strip()
        t     = self.cmb_type.currentText()
        if not start or not end:
            QMessageBox.warning(self, "Ошибка", "Заполните начало и конец события.")
            return
        ev = f"{t}: {start} - {end}"
        self.events.append((t, start, end))
        self.lst_events.addItem(ev)
        self.txt_start.clear()
        self.txt_end.clear()

    def run_model(self):
        model_name = self.cmb_model.currentText()
        path = MODELS[model_name]
        if not os.path.exists(path):
            QMessageBox.critical(self, "Ошибка", f"Модель {path} не найдена")
            return

        spec = importlib.util.spec_from_file_location("mod", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        df_events = self.create_df()

        try:
            if model_name == "Prophet":
                train_df, test_df = m.forecast_prophet(df_events)
                self._plot_prophet(train_df, test_df)

            elif model_name == "LightGBM":
                y, y_forecast = m.forecast_lgb(df_events)
                self._plot_lgb(y, y_forecast)

            else:
                pred_df,df = m.forecast_seq2seq(df_events)
                self._plot_generic(pred_df,df)

        except Exception as ex:
            QMessageBox.critical(self, "Ошибка при выполнении модели", str(ex))

    def _plot_prophet(self, train, test):
        self.ax.clear()
        self.ax.plot(train['Date'], train['Freight_Price'], label='Тренировочные', color='blue')
        self.ax.plot(test['Date'], test['Freight_Price'], label='Тест', color='blue')
        self.ax.plot(test['Date'], test['yhat_exp'], '--', label='Прогноз', color='red')
        self._finalize()

    def _plot_lgb(self, y, y_forecast):
        self.ax.clear()
        self.ax.plot(y.index, y.values, label='BDTI')
        if y_forecast is not None:
            self.ax.plot(y_forecast.index, y_forecast.values, '--', label='Прогноз BDTI', color='red')
        self._finalize()

    def _plot_generic(self, pred_df, df):
        self.ax.clear()
        self.ax.plot(pred_df, label='Прогноз', color='red', linestyle='--')
        self.ax.plot(df['Freight_Price'], label='Тренировочные', color='blue')
        self._finalize()

    def _finalize(self):
        self.ax.grid(True)
        self.ax.legend()
        for lbl in self.ax.get_xticklabels(): lbl.set_rotation(45)
        self.fig.tight_layout()
        self.canvas.draw()

    def create_df(self) -> pd.DataFrame:
        oil_source = self.cmb_oil_source.currentText()
        if oil_source == 'walletinvestor':
            oil = pd.read_csv("data/Updated_Oil_Price_Forecast.csv")
        else:
            oil = pd.read_csv("data/Updated_LiteFinance_Oil_Price_Forecast.csv")

        date_range = pd.date_range(start="2025-04-01", end="2030-05-01", freq="MS")
        df_events = pd.DataFrame({
            "Date":       date_range,
            "has_war":    0,
            "has_crisis": 0
        })
        for t, start, end in self.events:
            s = pd.to_datetime(start)
            e = pd.to_datetime(end)
            if t == "Кризис":
                df_events.loc[(df_events['Date'] >= s) & (df_events['Date'] <= e), "has_crisis"] = 1
            else:
                df_events.loc[(df_events['Date'] >= s) & (df_events['Date'] <= e), "has_war"]     = 1

        oil['Date'] = pd.to_datetime(oil['Date'])
        df_combined = pd.merge(oil, df_events, on="Date", how="left").fillna(0)
        return df_combined

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = ForecastWindow()
    w.showMaximized()
    sys.exit(app.exec_())
