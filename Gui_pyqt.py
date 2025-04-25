import sys
import os
import importlib.util
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QListWidget,
    QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd

# Список моделей
MODELS = {
    "Prophet": "Ignat_prophet/Main.py",
    "Seq2seq": "belG/lgbt.py",
    "LightGBM": "arsen/gb.py"
}
EVENT_TYPES = ["Кризис", "Война"]

class ForecastWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("График прогноза фрахта")
        self.events = []
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        main_v = QVBoxLayout(central)

        # Убираем заголовок, чтобы сгладить интерфейс
# (удален блок lbl_title)

        # график
        # Создаем фигуру и ось с увеличенным размером
        tmp_figsize = (12, 6)
        self.fig, self.ax = plt.subplots(figsize=tmp_figsize)
        self.canvas = FigureCanvas(self.fig)
        main_v.addWidget(self.canvas)
        # Дополнительный отступ между графиком и контролами
        main_v.addSpacing(40)
        # Добавляем отступ для снижения плотности интерфейса
        main_v.addSpacing(20)

        # контролы
        controls = QHBoxLayout()

        # модель
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(MODELS.keys())
        controls.addWidget(QLabel("Модель:"))
        controls.addWidget(self.cmb_model)

        # начало/конец
        self.txt_start = QLineEdit(); self.txt_start.setPlaceholderText("Начало")
        self.txt_end = QLineEdit(); self.txt_end.setPlaceholderText("Конец")
        controls.addWidget(self.txt_start)
        controls.addWidget(self.txt_end)

        # тип события + добавить
        self.cmb_type = QComboBox(); self.cmb_type.addItems(EVENT_TYPES)
        btn_add = QPushButton("Добавить событие")
        btn_add.clicked.connect(self.add_event)
        controls.addWidget(self.cmb_type)
        controls.addWidget(btn_add)

        # события
        self.lst_events = QListWidget()
        self.lst_events.setMaximumHeight(60)
        controls.addWidget(self.lst_events)

        # запуск
        btn_run = QPushButton("Запустить прогноз")
        btn_run.clicked.connect(self.run_model)
        controls.addWidget(btn_run)

        main_v.addLayout(controls)
        self.setCentralWidget(central)

    def add_event(self):
        start = self.txt_start.text().strip()
        end = self.txt_end.text().strip()
        t = self.cmb_type.currentText()
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

        try:
            if model_name == "Prophet":
                events_dict = {}
                for t,s,e in self.events: events_dict.setdefault(t, []).append((s,e))
                train_df, test_df = m.forecast_prophet(events_dict)
                self._plot_prophet(train_df, test_df)
            elif model_name == "LightGBM":
                events_dict = {}
                for t,s,e in self.events: events_dict.setdefault(t, []).append((s,e))
                y_train, y_test, y_pred = m.forecast_lgb(events_dict)
                self._plot_lgb(y_train, y_test, y_pred)
            else:
                for t,s,e in self.events: events_dict.setdefault(t, []).append((s,e))
                pred_df,df = m.forecast_seq2seq(self.events)
                self._plot_generic(pred_df,df)
        except Exception as ex:
            QMessageBox.critical(self, "Ошибка при выполнении модели", str(ex))

    def _plot_prophet(self, train, test):
        self.ax.clear()
        self.ax.plot(train['Date'], train['Freight_Price'], label='Тренировочные', color='blue')
        self.ax.plot(test['Date'], test['Freight_Price'], label='Тест', color='blue')
        self.ax.plot(test['Date'], test['yhat_exp'], '--', label='Прогноз', color='red')
        self._finalize()

    def _plot_lgb(self, y_train, y_test, y_pred):
        self.ax.clear()
        self.ax.plot(y_train.index, y_train.values, label='Тренировочные', color='blue')
        self.ax.plot(y_test.index, y_test.values, label='Тест', color='blue')
        self.ax.plot(y_test.index, y_pred, '--', label='Прогноз', color='red')
        self._finalize()

    def _plot_generic(self, pred_df, df):
        self.ax.clear()
        self.ax.plot(pred_df, label='Прогноз', color='red', linestyle='--')
        self.ax.plot(df['Freight_Price'], label='Истинное значение', color='blue')
        self._finalize()

    def _finalize(self):
        self.ax.grid(True)
        self.ax.legend()
        for lbl in self.ax.get_xticklabels(): lbl.set_rotation(45)
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = ForecastWindow()
    w.showMaximized()
    sys.exit(app.exec_())
