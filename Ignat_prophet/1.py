import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
bcti_df = pd.read_csv('//Users/ignat/Desktop/Demand/demand_forecasting/data/Baltic Clean Tanker Historical Data-2.csv', parse_dates=['Date'])
oil_df = pd.read_csv('/Users/ignat/Desktop/Demand/demand_forecasting/data/monthly_oil_cost_1988-2025.csv', parse_dates=['Date'])

# Сортировка по дате
bcti_df = bcti_df.sort_values('Date')
oil_df = oil_df.sort_values('Date')

# Переименование колонок для наглядности
bcti_df = bcti_df[['Date', 'Price']].rename(columns={'Price': 'BCTI_Price'})
oil_df = oil_df[['Date', 'Price']].rename(columns={'Price': 'Oil_Price'})

# Объединение по дате
merged_df = pd.merge(bcti_df, oil_df, on='Date', how='inner')
# Проверим типы данных в колонках, чтобы выявить проблему
merged_df.dtypes

import matplotlib.pyplot as plt

# График BCTI
plt.figure(figsize=(14, 5))
plt.plot(merged_df['Date'], merged_df['BCTI_Price'], color='orange', linewidth=2)
plt.title('Baltic Clean Tanker Index (BCTI) во времени')
plt.xlabel('Дата')
plt.ylabel('Индекс BCTI')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# График цены на нефть
plt.figure(figsize=(14, 5))
plt.plot(merged_df['Date'], merged_df['Oil_Price'], color='green', linewidth=2)
plt.title('Цена на нефть во времени')
plt.xlabel('Дата')
plt.ylabel('Цена нефти (USD)')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

