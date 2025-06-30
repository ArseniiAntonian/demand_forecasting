import sys
import os
import importlib.util
from datetime import datetime
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QListWidgetItem,
    QDateEdit, QListWidget, QDoubleSpinBox, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd

MODELS = {
    "Prophet":   "Ignat_prophet/NEW_predict.py",
    "Seq2seq w exogs": "belG/tcn_forecast.py",
    "LightGBM":  "arsen/gb_forecast.py"
}
CRISIS_TYPES = [
    "Financial",
    "Pandemic",
    "Geopolitical",
    "Natural",
    "Logistical",
]
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

        # График
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        main_v.addWidget(self.canvas)

        # Первый ряд контролов
        controls1 = QHBoxLayout()
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(MODELS.keys())
        controls1.addWidget(QLabel("Модель:"))
        controls1.addWidget(self.cmb_model)

        self.cmb_oil_source = QComboBox()
        self.cmb_oil_source.addItems(OIL_FORECAST_TYPES)
        controls1.addWidget(QLabel("Источник нефти:"))
        controls1.addWidget(self.cmb_oil_source)

        self.txt_start = QDateEdit()
        self.txt_start.setDisplayFormat("yyyy-MM")
        self.txt_start.setCalendarPopup(True)
        self.txt_start.setMinimumDate(QDate(1900,1,1))
        self.txt_start.setMaximumDate(QDate(2100,12,1))
        self.txt_start.setDate(QDate.currentDate())
        controls1.addWidget(QLabel("Начало (YYYY-MM):"))
        controls1.addWidget(self.txt_start)

        self.txt_end = QDateEdit()
        self.txt_end.setDisplayFormat("yyyy-MM")
        self.txt_end.setCalendarPopup(True)
        self.txt_end.setMinimumDate(QDate(1900,1,1))
        self.txt_end.setMaximumDate(QDate(2100,12,1))
        self.txt_end.setDate(QDate.currentDate())
        controls1.addWidget(QLabel("Конец (YYYY-MM):"))
        controls1.addWidget(self.txt_end)

        main_v.addLayout(controls1)

        # Второй ряд контролов
        controls2 = QHBoxLayout()
        self.cmb_type = QComboBox()
        self.cmb_type.addItems(CRISIS_TYPES)
        controls2.addWidget(QLabel("Тип кризиса:"))
        controls2.addWidget(self.cmb_type)

        self.spin_intensity = QDoubleSpinBox()
        self.spin_intensity.setRange(0.0, 1.0)
        self.spin_intensity.setSingleStep(0.1)
        self.spin_intensity.setValue(0.5)
        controls2.addWidget(QLabel("Интенсивность:"))
        controls2.addWidget(self.spin_intensity)

        self.chk_shock = QCheckBox("Шоковый кризис")
        controls2.addWidget(self.chk_shock)

        btn_add = QPushButton("Добавить событие")
        btn_add.clicked.connect(self.add_event)
        controls2.addWidget(btn_add)

        self.lst_events = EventListWidget()
        self.lst_events.setMaximumHeight(80)
        controls2.addWidget(self.lst_events)

        btn_run = QPushButton("Запустить прогноз")
        btn_run.clicked.connect(self.run_model)
        controls2.addWidget(btn_run)

        main_v.addLayout(controls2)

        self.setCentralWidget(central)

    def add_event(self):
        start_date = self.txt_start.date().toPyDate()
        end_date   = self.txt_end.date().toPyDate()
        if end_date < start_date:
            return

        t = self.cmb_type.currentText()
        intensity = float(self.spin_intensity.value())
        shock = 1 if self.chk_shock.isChecked() else 0

        txt = f"{t}: {start_date.strftime('%Y-%m')}–{end_date.strftime('%Y-%m')} " \
              f"(int={intensity:.1f}, shock={shock})"
        item = QListWidgetItem(txt)
        item.setData(Qt.UserRole, (t, start_date, end_date, intensity, shock))
        self.lst_events.addItem(item)
        self.events.append((t, start_date, end_date, intensity, shock))

        today = QDate.currentDate()
        self.txt_start.setDate(today)
        self.txt_end.setDate(today)
        self.spin_intensity.setValue(0.5)
        self.chk_shock.setChecked(False)

    def run_model(self):
        spec = importlib.util.spec_from_file_location("mod", MODELS[self.cmb_model.currentText()])
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        df_events = self.create_df()
        if self.cmb_model.currentText() == "Prophet":
            train, test = m.forecast_prophet(df_events)
            self._plot_prophet(train, test)
        elif self.cmb_model.currentText() == "LightGBM":
            y, y_pred = m.forecast_lgb(df_events)
            self._plot_lgb(y, y_pred)
        else:
            pred, df_hist = m.forecast_last_data_w_exogs(df_events)
            self._plot_generic(pred, df_hist)

    def _plot_prophet(self, train, test):
        self.ax.clear()
        self.ax.plot(train['Date'], train['Freight_Price'], label='BDTI')
        self.ax.plot(test['ds'], test['yhat_exp'], '--', label='Прогноз')
        self._finalize()

    def _plot_lgb(self, y, y_forecast):
        self.ax.clear()
        self.ax.plot(y.index, y.values, label='BDTI')
        if y_forecast is not None:
            self.ax.plot(y_forecast.index, y_forecast.values, '--', label='Прогноз')
        self._finalize()

    def _plot_generic(self, pred, df):
        self.ax.clear()
        self.ax.plot(pred, '--', label='Прогноз')
        self.ax.plot(df['Freight_Price'], label='История')
        self._finalize()

    def _finalize(self):
        self.ax.grid(True)
        self.ax.legend()
        for lbl in self.ax.get_xticklabels(): lbl.set_rotation(45)
        self.fig.tight_layout()
        self.canvas.draw()

    def create_df(self) -> pd.DataFrame:
        oil_src = self.cmb_oil_source.currentText()
        path = (
            "data/Updated_Oil_Price_Forecast.csv"
            if oil_src == 'walletinvestor'
            else "data/Updated_LiteFinance_Oil_Price_Forecast.csv"
        )
        oil = pd.read_csv(path)
        oil['Date'] = pd.to_datetime(oil['Date'])
        dates = pd.date_range("2025-04-01", "2030-05-01", freq="MS")
        df_ev = pd.DataFrame({'Date': dates})
        df_ev['has_crisis'] = 0
        df_ev['crisis_intensity'] = 0.0
        df_ev['crisis_shock'] = 0
        for ct in CRISIS_TYPES:
            df_ev[f'crisis_type_{ct}'] = 0
        import numpy as np
        intensity_sum = np.zeros(len(df_ev), dtype=float)
        count = np.zeros(len(df_ev), dtype=int)
        for t, s, e, intensity, shock in self.events:
            mask = (df_ev['Date'] >= pd.to_datetime(s)) & (df_ev['Date'] <= pd.to_datetime(e))
            df_ev.loc[mask, 'has_crisis'] = 1
            df_ev.loc[mask, f'crisis_type_{t}'] = 1
            intensity_sum[mask] += intensity
            count[mask] += 1
            df_ev.loc[mask, 'crisis_shock'] = np.maximum(df_ev.loc[mask, 'crisis_shock'], shock)
        with np.errstate(divide='ignore', invalid='ignore'):
            df_ev['crisis_intensity'] = np.where(count>0, intensity_sum/count, 0.0)
        df_full = pd.merge(oil, df_ev, on='Date', how='left').fillna(0)
        return df_full

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = ForecastWindow()
    w.showMaximized()
    sys.exit(app.exec_())
