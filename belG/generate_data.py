from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

def generate_synthetic_tail(df,
                            date_col: str,
                            value_cols: list,
                            n_periods: int,
                            freq: str = 'M',
                            spline_trend: bool = False,
                            random_seed: int = 42):
    """
    df           — исходный DataFrame с колонками date_col и value_cols
    date_col     — название столбца с датой
    value_cols   — список имён числовых колонок, которые хотим удлинить (например ['Oil','Freight'])
    n_periods    — сколько точек «в прошлое» сгенерировать
    freq         — частота (например 'M' для месячных точек)
    spline_trend — если True, тренд интерполируется кубическим сплайном (иначе — линейно)
    """
    np.random.seed(random_seed)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    synthetic = {}
    for col in value_cols:
        # 1. STL-разложение
        stl = STL(df[col], period=12, robust=True)  # годовая сезонность
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid.dropna()
        
        # 2. вычленяем линейный тренд (или сплайн)
        if not spline_trend:
            # приближение линейным регрессором
            idx_num = np.arange(len(trend))
            coefs = np.polyfit(idx_num, trend.values, 1)
            trend_line = pd.Series(np.polyval(coefs, idx_num),
                                   index=trend.index)
        else:
            trend_line = trend.interpolate('cubic')
        
        # 3. строим календарь назад
        last_date = df.index.min()
        new_dates = pd.date_range(end=last_date - pd.Timedelta(1, 'D'),
                                  periods=n_periods, freq=freq)
        
        # 4. бутстрэпим остатки и сезонность
        resid_pool = resid.values
        seasonal_cycle = seasonal[-12:].values  # последний год сезонности
        
        gen = []
        for i, dt in enumerate(new_dates[::-1]):  # сдвиг назад
            # индекс «назад» в сезонном цикле
            season = seasonal_cycle[i % 12]
            # линейно экстраполированный тренд
            # берем трендовые coefs, но для отрицательного idx
            idx_back = - (i + 1)
            if not spline_trend:
                trend_val = np.polyval(coefs, idx_back)
            else:
                # если сплайн, просто возьмём первый трендовый элемент
                trend_val = trend_line.iloc[0]
            # бутстрэп остатка
            resid_val = np.random.choice(resid_pool)
            gen.append(trend_val + season + resid_val)
        
        # соберём сгенерированные значения в Series
        synthetic[col] = pd.Series(gen[::-1], index=new_dates)
    
    # 5. склеиваем
    df_synth = pd.DataFrame(synthetic)
    full = pd.concat([df_synth, df], axis=0).sort_index()
    
    # 6. пересчёт лагов (пример для лагов 1…3 по обоим рядам)
    for col in value_cols:
        for lag in [1, 2, 3]:
            full[f'{col}_lag{lag}'] = full[col].shift(lag)
    
    full.reset_index(inplace=True)
    full.rename(columns={'index': date_col}, inplace=True)
    return full

# === пример использования ===
if __name__ == '__main__':
    # 1) читаем ваш CSV
    df = pd.read_csv('data/ML_filled.csv')  
    # 2) генерим 60 «старых» точек назад по месяцам
    df.plot(x='Date', y='Freight_Price')
    plt.show()
