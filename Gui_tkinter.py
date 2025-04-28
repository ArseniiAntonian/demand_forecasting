import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import importlib.util
import os
import pandas as pd

# Список моделей
MODELS = {
    "Prophet": "Ignat_prophet/Main.py",
    "Seq2seq": "belG/lgbt.py",
    "LightGBM": "arsen/gb.py"
}

# Типы событий
EVENT_TYPES = ["Кризис", "Война"]

class FreightForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("График прогноза фрахта")

        # Храним события как список кортежей (тип, начало, конец)
        self.events = []

        # Создаем фигуру и ось заранее
        self.fig, self.ax = plt.subplots(figsize=(10, 5))

        self.setup_ui()

    def setup_ui(self):
        # Заголовок
        title = tk.Label(self.root, text="ГРАФИК", font=("Arial", 20))
        title.pack(pady=5)

        # Нижняя панель для управления (фиксированная высота)
        bottom_frame = tk.Frame(self.root, height=120)
        bottom_frame.pack_propagate(False)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Используем grid для компактного расположения
        for col in range(6):
            bottom_frame.columnconfigure(col, weight=1)

        # --- Выбор модели ---
        tk.Label(bottom_frame, text="Модель:").grid(row=0, column=0, sticky='w', padx=5)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            bottom_frame,
            textvariable=self.model_var,
            values=list(MODELS.keys()),
            state='readonly',
            width=15
        )
        self.model_dropdown.grid(row=1, column=0, padx=5, sticky='w')

        # --- Поля начала и конца ---
        tk.Label(bottom_frame, text="Начало:").grid(row=0, column=1, sticky='w', padx=5)
        self.start_entry = tk.Entry(bottom_frame, width=12)
        self.start_entry.grid(row=1, column=1, padx=5, sticky='w')

        tk.Label(bottom_frame, text="Конец:").grid(row=0, column=2, sticky='w', padx=5)
        self.end_entry = tk.Entry(bottom_frame, width=12)
        self.end_entry.grid(row=1, column=2, padx=5, sticky='w')

        # --- Выбор типа события и кнопка добавить ---
        tk.Label(bottom_frame, text="Тип события:").grid(row=0, column=3, sticky='w', padx=5)
        self.event_type_var = tk.StringVar()
        self.event_type_dropdown = ttk.Combobox(
            bottom_frame,
            textvariable=self.event_type_var,
            values=EVENT_TYPES,
            state='readonly',
            width=12
        )
        self.event_type_dropdown.grid(row=1, column=3, padx=5, sticky='w')
        tk.Button(
            bottom_frame, text="Добавить событие", command=self.add_event,
            width=15
        ).grid(row=1, column=4, padx=5, sticky='w')

        # --- Список событий ---
        tk.Label(bottom_frame, text="События:").grid(row=0, column=5, sticky='w', padx=5)
        self.events_listbox = tk.Listbox(bottom_frame, width=30, height=4)
        self.events_listbox.grid(row=1, column=5, padx=5, sticky='w')

        # --- Кнопка запуска ---
        tk.Button(
            bottom_frame, text="Запустить прогноз", command=self.run_model,
            width=20
        ).place(relx=0.5, rely=0.5, anchor='s')

        # Область для графика пакуем после панели управления
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_event(self):
        start = self.start_entry.get().strip()
        end = self.end_entry.get().strip()
        e_type = self.event_type_var.get().strip()
        if not (start and end and e_type):
            messagebox.showwarning("Ошибка", "Заполните все поля события")
            return
        event_str = f"{e_type}: {start} - {end}"
        self.events.append((e_type, start, end))
        self.events_listbox.insert(tk.END, event_str)
        self.start_entry.delete(0, tk.END)
        self.end_entry.delete(0, tk.END)

    def run_model(self):
        selected_model = self.model_var.get()
        if not selected_model:
            messagebox.showwarning("Ошибка", "Выберите модель")
            return
        model_path = MODELS[selected_model]
        if not os.path.exists(model_path):
            messagebox.showerror("Ошибка", f"Модель {model_path} не найдена")
            return
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        try:
            if selected_model == "Prophet":
                events_dict = {}
                for e_type, start, end in self.events:
                    events_dict.setdefault(e_type, []).append((start, end))
                train_df, test_df = model_module.forecast_prophet(events_dict)
                # Рисуем
                self.ax.clear()
                self.ax.plot(train_df['Date'], train_df['Freight_Price'], label='Тренировочные данные')
                self.ax.plot(test_df['Date'], test_df['Freight_Price'], label='Тест (2023–2025)')
                self.ax.plot(test_df['Date'], test_df['yhat_exp'], '--', label='Прогноз')
                self.ax.set_title("Прогноз фрахта: Prophet + внешние признаки (2023–2025)")
                self.ax.set_xlabel("Дата")
                self.ax.set_ylabel("Фрахтовая ставка")
                self.ax.legend()
                for lbl in self.ax.get_xticklabels(): lbl.set_rotation(45)
                self.fig.tight_layout()
                self.canvas.draw()

            elif selected_model == "LightGBM":
                # Формируем словарь событий
                events_dict = {}
                for e_type, start, end in self.events:
                    events_dict.setdefault(e_type, []).append((start, end))
                # теперь forecast_lgb возвращает y_train, y_test, y_pred
                y_train, y_test, y_pred = model_module.forecast_lgb(events_dict)

                # Очищаем ось
                self.ax.clear()

                # Actual – тренировочные
                self.ax.plot(
                    y_train.index, y_train.values,
                    label='Тренировочные данные'
                )
                # Actual – тестовые
                self.ax.plot(
                    y_test.index, y_test.values,
                    label='Тест (последние 24)', 
                    color='green'
                )
                # Прогноз
                self.ax.plot(
                    y_test.index, y_pred,
                    '--', label='Прогноз', color='red'
                )

                # Сетка, подписи, легенда
                self.ax.grid(True)
                self.ax.set_title('Предсказания LightGBM модели')
                self.ax.set_xlabel('Дата')
                self.ax.set_ylabel('Цена фрахта')
                self.ax.legend()
                for lbl in self.ax.get_xticklabels():
                    lbl.set_rotation(45)
                self.fig.tight_layout()
                self.canvas.draw()

            else:
                x, y = model_module.run_forecast(self.events)
                try:
                    x = list(x)
                    y = list(y)
                except:
                    x = list(x.index)
                    y = list(y)
                if len(x) != len(y):
                    messagebox.showerror("Ошибка данных", f"len(x)={len(x)}, len(y)={len(y)} — нельзя построить график")
                    return
                self.ax.clear()
                self.ax.plot(x, y, label=selected_model)
                self.ax.set_title("Прогноз фрахта")
                self.ax.legend()
                self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Ошибка при выполнении модели", str(e))

if __name__ == '__main__':
    root = tk.Tk()
    app = FreightForecastApp(root)
    root.mainloop()
