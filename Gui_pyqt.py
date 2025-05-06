import sys
import os
import importlib.util
from datetime import datetime
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QListWidgetItem,
    QMessageBox, QDateEdit, QListWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd

MODELS = {
    "Prophet":   "Ignat_prophet/1.py",
    "Seq2seq":   "belG/lgbt.py",
    "LightGBM":  "arsen/gb_forecast.py"
}
EVENT_TYPES = ["Кризис", "Война"]
OIL_FORECAST_TYPES = ["walletinvestor", "EFA forecast"]

class EventListWidget(QListWidget):
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            window = self.window()
            for item in self.selectedItems():
                data = item.data(Qt.UserRole)
                if data in window.events:
                    window.events.remove(data)
                self.takeItem(self.row(item))
        else:
            super().keyPressEvent(event)

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

        controls = QHBoxLayout()

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(MODELS.keys())
        controls.addWidget(QLabel("Модель:"))
        controls.addWidget(self.cmb_model)

        self.cmb_oil_source = QComboBox()
        self.cmb_oil_source.addItems(OIL_FORECAST_TYPES)
        controls.addWidget(QLabel("Источник нефти:"))
        controls.addWidget(self.cmb_oil_source)

        self.txt_start = QDateEdit()
        self.txt_start.setDisplayFormat("yyyy-MM")
        self.txt_start.setCalendarPopup(True)
        self.txt_start.setMinimumDate(QDate(1900,1,1))
        self.txt_start.setMaximumDate(QDate(2100,12,1))
        self.txt_start.setDate(QDate.currentDate())

        self.txt_end = QDateEdit()
        self.txt_end.setDisplayFormat("yyyy-MM")
        self.txt_end.setCalendarPopup(True)
        self.txt_end.setMinimumDate(QDate(1900,1,1))
        self.txt_end.setMaximumDate(QDate(2100,12,1))
        self.txt_end.setDate(QDate.currentDate())

        controls.addWidget(QLabel("Начало (YYYY-MM):"))
        controls.addWidget(self.txt_start)
        controls.addWidget(QLabel("Конец (YYYY-MM):"))
        controls.addWidget(self.txt_end)

        self.cmb_type = QComboBox()
        self.cmb_type.addItems(EVENT_TYPES)
        btn_add = QPushButton("Добавить событие")
        btn_add.clicked.connect(self.add_event)
        controls.addWidget(self.cmb_type)
        controls.addWidget(btn_add)

        self.lst_events = EventListWidget()
        self.lst_events.setMaximumHeight(80)
        controls.addWidget(self.lst_events)

        btn_run = QPushButton("Запустить прогноз")
        btn_run.clicked.connect(self.run_model)
        controls.addWidget(btn_run)

        main_v.addLayout(controls)
        self.setCentralWidget(central)

    def add_event(self):
        start_date = self.txt_start.date().toPyDate()
        end_date   = self.txt_end.date().toPyDate()

        if end_date < start_date:
            QMessageBox.warning(self, "Ошибка", "Дата окончания раньше даты начала.")
            return

        t = self.cmb_type.currentText()
        s_str = start_date.strftime("%Y-%m")
        e_str = end_date.strftime("%Y-%m")
        text = f"{t}: {s_str} – {e_str}"

        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, (t, start_date, end_date))
        self.lst_events.addItem(item)
        self.events.append((t, start_date, end_date))

        # сбросить на сегодня
        today = QDate.currentDate()
        self.txt_start.setDate(today)
        self.txt_end.setDate(today)

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
                train, test = m.forecast_prophet(df_events)
                self._plot_prophet(train, test)
            elif model_name == "LightGBM":
                y, y_pred = m.forecast_lgb(df_events)
                self._plot_lgb(y, y_pred)
            else:
                pred, df = m.forecast_seq2seq(df_events)
                self._plot_generic(pred, df)
        except Exception as ex:
            QMessageBox.critical(self, "Ошибка при выполнении модели", str(ex))

    def _plot_prophet(self, train, test):
        self.ax.clear()
        self.ax.plot(train['Date'], train['Freight_Price'], label='BDTI', color='blue')
        # self.ax.plot(test['Date'], test['Freight_Price'], label='Тест')
        self.ax.plot(test['ds'], test['yhat_exp'], '--', label='Прогноз BDTI', color='red')
        self._finalize()

    def _plot_lgb(self, y, y_forecast):
        self.ax.clear()
        self.ax.plot(y.index, y.values, label='BDTI', color='blue')
        if y_forecast is not None:
            self.ax.plot(y_forecast.index, y_forecast.values, '--', label='Прогноз BDTI', color='red')
        self._finalize()

    def _plot_generic(self, pred, df):
        self.ax.clear()
        self.ax.plot(pred, '--', label='Прогноз BDTI', color='red')
        self.ax.plot(df['Freight_Price'], label='BDTI', color='blue')
        self._finalize()

    def _finalize(self):
        self.ax.grid(True)
        self.ax.legend()
        for lbl in self.ax.get_xticklabels():
            lbl.set_rotation(45)
        self.fig.tight_layout()
        self.canvas.draw()

    def create_df(self) -> pd.DataFrame:
        oil_src = self.cmb_oil_source.currentText()
        path = "data/Updated_Oil_Price_Forecast.csv" if oil_src=='walletinvestor' \
               else "data/Updated_LiteFinance_Oil_Price_Forecast.csv"
        oil = pd.read_csv(path)
        oil['Date'] = pd.to_datetime(oil['Date'])

        dates = pd.date_range("2025-04-01", "2030-05-01", freq="MS")
        df_ev = pd.DataFrame({
            "Date":       dates,
            "has_war":    0,
            "has_crisis": 0
        })

        for t, s_date, e_date in self.events:
            s = pd.to_datetime(s_date)
            e = pd.to_datetime(e_date)
            mask = (df_ev['Date'] >= s) & (df_ev['Date'] <= e)
            if t == "Кризис":
                df_ev.loc[mask, "has_crisis"] = 1
            else:
                df_ev.loc[mask, "has_war"] = 1

        return pd.merge(oil, df_ev, on="Date", how="left").fillna(0)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = ForecastWindow()
    w.showMaximized()
    sys.exit(app.exec_())
