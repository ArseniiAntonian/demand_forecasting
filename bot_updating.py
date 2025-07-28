import logging
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import date
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
import os

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    ConversationHandler, MessageHandler, filters, ContextTypes
)
from telegram_bot_calendar import DetailedTelegramCalendar

from Ignat_prophet.NEW_predict import forecast_prophet
from belG.tcn_forecast import forecast_last_data_w_exogs
from arsen.gb_forecast import forecast_lgb

logging.basicConfig(level=logging.INFO)
BOT_TOKEN = "8144080240:AAEZelQNgc-OGGp-vdyQKT7eho_nKAWx9j4"

(
    SELECT_MODEL,
    SELECT_OIL_SOURCE,
    SELECT_PREDEFINED_CRISES,
    ASK_NUM_CRISES,
    SELECT_CRISIS_TYPE,
    CALENDAR_START,
    CALENDAR_END,
    INPUT_SHOCK,
    INPUT_INTENSITY,
    AFTER_FORECAST,
    RANGE_START,
    RANGE_END,
    FEEDBACK
) = range(13)

MODEL_OPTIONS = {
    'prophet': ('Модель 1',    forecast_prophet),
    'tcn':     ('Модель 2',        forecast_last_data_w_exogs),
    'gb':      ('Модель 3',   forecast_lgb),
}

CRISIS_TYPES = {
    'Financial':   'Финансовый',
    'Pandemic':    'Пандемический',
    'Geopolitical':'Геополитический',
    'Natural':     'Природный',
    'Logistical':  'Логистический',
}

PREDEFINED_CRISES = [
    {
        'label': 'Конфликт Россия–Украина',
        'type': 'Geopolitical',
        'start': '2022-03-01',
        'end':   '2030-12-01',
        'intensity': 0.8,
        'shock': 1,
        'description': 'Санкции, срыв транзита и рост страховых тарифов.'
    },
    {
        'label': 'Напряжённость Китай–Тайвань',
        'type': 'Geopolitical',
        'start': '2023-01-01',
        'end':   '2030-12-01',
        'intensity': 0.7,
        'shock': 1,
        'description': 'Блокировка морских коридоров, суда вынуждены идти в обход.'
    },
    {
        'label': 'Климатические экстремумы',
        'type': 'Natural',
        'start': '2025-01-01',
        'end':   '2030-12-01',
        'intensity': 0.9,
        'shock': 0,
        'description': 'Ураганы, штормы и засухи приводят к сбоям в портах.'
    },
    {
        'label': 'Узкие места энергоперехода',
        'type': 'Financial',
        'start': '2025-01-01',
        'end':   '2028-12-01',
        'intensity': 0.7,
        'shock': 0,
        'description': 'Дефицит лития и кобальта — волатильность цен на топливо.'
    },
    {
        'label': 'Дисрупции цепочек поставок',
        'type': 'Logistical',
        'start': '2025-01-01',
        'end':   '2030-12-01',
        'intensity': 0.6,
        'shock': 0,
        'description': 'Тарифы и квоты нарушают привычные маршруты судов.'
    },
    {
        'label': 'Новый глобальный финансовый кризис',
        'type': 'Financial',
        'start': '2026-01-01',
        'end':   '2028-12-01',
        'intensity': 0.8,
        'shock': 1,
        'description': '«Пузыри» на рынках и отток капитала — сжатие торговли.'
    },
    {
        'label': 'Новый пандемический шок',
        'type': 'Pandemic',
        'start': '2027-01-01',
        'end':   '2027-08-01',
        'intensity': 0.7,
        'shock': 1,
        'description': 'Вспышки вируса, локдауны и сокращение рабочих смен.'
    },
]

# Entry point
def start_states():
    return {
        SELECT_MODEL:              [CallbackQueryHandler(model_chosen)],
        SELECT_OIL_SOURCE:         [CallbackQueryHandler(oil_source_chosen)],
        SELECT_PREDEFINED_CRISES:  [CallbackQueryHandler(predefined_crises_chosen)],
        ASK_NUM_CRISES:            [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_num_crises)],
        SELECT_CRISIS_TYPE:        [CallbackQueryHandler(select_crisis_type)],
        CALENDAR_START:            [CallbackQueryHandler(calendar_start)],
        CALENDAR_END:              [CallbackQueryHandler(calendar_end)],
        INPUT_SHOCK:              [CallbackQueryHandler(shock_chosen, pattern='^shock_')],
        INPUT_INTENSITY:           [MessageHandler(filters.TEXT & ~filters.COMMAND, input_intensity)],
        AFTER_FORECAST:            [
            CallbackQueryHandler(repeat_forecast, pattern='^repeat$'),
            CallbackQueryHandler(range_start,    pattern='^range$'),
            CallbackQueryHandler(feedback_start, pattern='^feedback$'),
            CallbackQueryHandler(exit_bot,       pattern='^exit$'),
        ],
        RANGE_START:               [CallbackQueryHandler(range_end)],
        RANGE_END:                 [CallbackQueryHandler(range_end)],
        FEEDBACK:                  [MessageHandler(filters.TEXT & ~filters.COMMAND, feedback_received)],
    }


# Entry point
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message or update.callback_query.message
    if update.callback_query:
        await update.callback_query.answer()
    welcome = (
        "Добро пожаловать в бот-прогнозист фрахтовых цен!\n\n"
        "Что умеет бот:\n"
        "  • Строит график фактических цен (2003–2025) и прогноз (2025–2030).\n"
        "  • Поддерживает три модели:\n" 
        "           1) Аддитивная модель временных рядов\n"
        "           2) Нейросетевая модель\n "
        "           3) Ансамблевый метод машинного обучения \n"
        "  • Учтёт ваши сценарии кризисов: тип, даты и интенсивность.\n\n"
        "Как начать работу:\n"
        "1️⃣ Шаг 1: Нажмите на кнопку с моделью, которую хотите использовать.\n"
        "2️⃣ Шаг 2: Выберите источник прогноза нефти — Brent или WTI.\n"
        "   • Brent — эталон для Европы и Азии, чувствителен к геополитике.\n"
        "   • WTI — ориентир для США, более волатилен и зависит от внутренней добычи.\n"
        "3️⃣ Шаг 3: Укажите, сколько кризисов заложить (0–10).\n"
        "   – Для каждого кризиса выберите:\n"
        "     • Тип кризиса (финансовый, пандемический и т. д.).\n"
        "     • Дату начала и окончания через календарь.\n"
        "     • Интенсивность в процентах (0–100%) — отражает силу и длительность воздействия.\n"
        "     • Шоковость (да/нет) — указывает, произошло ли событие резко или развивалось постепенно.\n\n"
        "4️⃣ Шаг 4: Дождитесь построения графика. После будут четыре кнопки:\n"
        "   • «Повторить прогноз» — начать всё сначала.\n"
        "   • «Выбрать интервал прогноза» — посмотреть прогноз за выбранный период.\n"
        "   • «Оставить отзыв» — поделиться впечатлениями.\n"
        "   • «Выход» — завершить работу бота.\n\n"
        "После прогноза вы сможете вернуться и снова выбрать модель, указать новые кризисы или просто выйти."
    )
    await msg.reply_text(welcome)
    kb = [[InlineKeyboardButton(name, callback_data=key)]
          for key,(name,_) in MODEL_OPTIONS.items()]
    await msg.reply_text('⏩ Выберите модель:', reply_markup=InlineKeyboardMarkup(kb))
    return SELECT_MODEL

async def model_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    # Удаляем исходное сообщение
    await q.message.delete()

    # Сохраняем и эхо
    context.user_data['model_key'] = q.data
    name, _ = MODEL_OPTIONS[q.data]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Модель: *{name}*',
        parse_mode='Markdown'
    )

    # Кнопки для выбора нефти
    buttons = [
        InlineKeyboardButton('Brent', callback_data='brent'),
        InlineKeyboardButton('WTI',   callback_data='wti')
    ]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Выберите нефть:',
        reply_markup=InlineKeyboardMarkup([buttons])
    )
    return SELECT_OIL_SOURCE


async def oil_source_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    # Удаляем исходное сообщение
    await q.message.delete()

    # Сохраняем и эхо
    context.user_data['oil_source'] = q.data
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Нефть: *{q.data.upper()}*',
        parse_mode='Markdown'
    )

    # Строим список предопределённых кризисов
    lines = ['*Выберите кризисы:*']
    kb = []
    for idx, c in enumerate(PREDEFINED_CRISES, start=1):
        date_range = f'{c["start"][:7]}–{c["end"][:7]}'
        intensity = int(c['intensity'] * 100)
        shock = 'да' if c['shock'] else 'нет'
        # одна строка с основными параметрами
        lines.append(
            f'\n*Кризис {idx}*: {c["label"]} ({CRISIS_TYPES[c["type"]]}) — '
            f'{date_range}, инт. {intensity}%, шок: {shock}'
        )
        # вторая строка — описание
        lines.append(f'_{c["description"]}_')
        kb.append([InlineKeyboardButton(f'Кризис {idx}', callback_data=f'pre_{idx}')])

    # Инструкция
    lines.append(
        '\nПосле выбора кризисов нажмите кнопку *Готово*, '
        'или выберите *Ввести вручную*.'
    )
    kb.append([
        InlineKeyboardButton('Без кризисов',    callback_data='no_crises'),
        InlineKeyboardButton('Ввести вручную',  callback_data='manual_crises'),
        InlineKeyboardButton('Готово',          callback_data='pre_done'),
    ])

    # Отправляем всё единым сообщением
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='\n'.join(lines),
        reply_markup=InlineKeyboardMarkup(kb),
        parse_mode='Markdown'
    )
    return SELECT_PREDEFINED_CRISES



async def predefined_crises_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    d = q.data

    if d == 'no_crises':
        context.user_data['crises'] = []
        context.user_data['num_crises'] = 0
        return await launch_forecast(update, context)

    if d == 'manual_crises':
        await q.message.reply_text('Сколько кризисов (0–10)?')
        return ASK_NUM_CRISES

    if d == 'pre_done':
        sel = context.user_data.get('pre_selected', [])
        context.user_data['crises'] = [
            {
                'type': PREDEFINED_CRISES[i-1]['type'],
                'start': PREDEFINED_CRISES[i-1]['start'],
                'end':   PREDEFINED_CRISES[i-1]['end'],
                'intensity': PREDEFINED_CRISES[i-1]['intensity'],
                'shock':     PREDEFINED_CRISES[i-1]['shock']
            }
            for i in sel
        ]
        context.user_data['num_crises'] = len(sel)
        return await launch_forecast(update, context)

    # переключение выбора отдельных кризисов
    if d.startswith('pre_'):
        i = int(d.split('_')[1])
        sel = context.user_data.setdefault('pre_selected', [])
        if i in sel:
            sel.remove(i)
            await q.message.reply_text(f'Убран кризис {i}')
        else:
            sel.append(i)
            await q.message.reply_text(f'Добавлен кризис {i}')
        return SELECT_PREDEFINED_CRISES

async def ask_num_crises(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        n = int(update.message.text.strip())
        assert 0 <= n <= 10
    except:
        await update.message.reply_text('Введите число от 0 до 10.')
        return ASK_NUM_CRISES
    context.user_data['num_crises'] = n
    context.user_data['crises'] = []
    if n == 0:
        return await launch_forecast(update, context)
    context.user_data['current'] = 1
    kb = [[InlineKeyboardButton(label, callback_data=key)]
          for key, label in CRISIS_TYPES.items()]
    await update.message.reply_text(
        'Выберите тип кризиса #1:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_CRISIS_TYPE

async def select_crisis_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    # 1) Удаляем сообщение с кнопками
    await q.message.delete()

    # 2) Сохраняем выбор и выводим эхо
    ctype_key = q.data
    idx = context.user_data['current']
    context.user_data['crises'].append({'type': ctype_key})
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Тип кризиса #{idx}: *{CRISIS_TYPES[ctype_key]}*',
        parse_mode='Markdown'
    )

    # 3) Спрашиваем дату начала с помощью календаря
    cal, _ = DetailedTelegramCalendar(
        min_date=date(2025, 1, 1),
        max_date=date(2030, 12, 31)
    ).build()
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Выберите дату начала для кризиса #{idx} ({CRISIS_TYPES[ctype_key]}):',
        reply_markup=cal
    )
    return CALENDAR_START

async def calendar_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    cal_obj = DetailedTelegramCalendar(
        min_date=date(2025, 1, 1),
        max_date=date(2030, 12, 31)
    )
    result, cal, step = cal_obj.process(q.data)
    if result is None and cal:
        await q.edit_message_text(text=step, reply_markup=cal)
        return CALENDAR_START
    if result is None and cal is None:
        await context.bot.delete_message(chat_id=q.message.chat.id, message_id=q.message.message_id)
        cal2, _ = cal_obj.build()
        await q.message.reply_text('Неверный ввод даты начала. Выберите снова:', reply_markup=cal2)
        return CALENDAR_START

    # Сбрасываем день на 1 и сохраняем год-месяц
    result = result.replace(day=1)
    await q.edit_message_text(f'Дата начала: {result.strftime("%Y-%m")}')
    context.user_data['crises'][-1]['start'] = result.isoformat()

    # Дальше выбор даты окончания
    cal2, _ = cal_obj.build()
    await q.message.reply_text('Выберите месяц и год окончания:', reply_markup=cal2)
    return CALENDAR_END


async def range_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    cal_obj = DetailedTelegramCalendar(
        min_date=date(2025, 1, 1),
        max_date=date(2030, 12, 31)
    )
    result, cal, step = cal_obj.process(q.data)

    # Листаем календарь
    if result is None and cal:
        await q.edit_message_text(text=step, reply_markup=cal)
        return RANGE_END

    # Неверный ввод
    if result is None and cal is None:
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=context.user_data.pop('range_msg_id')
        )
        cal2, _ = cal_obj.build()
        msg = await q.message.reply_text(
            '❗ Неверный ввод. Попробуйте снова.',
            reply_markup=cal2
        )
        context.user_data['range_msg_id'] = msg.message_id
        return RANGE_END

    # Фиксация начала/конца
    if 'range_start' not in context.user_data:
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=context.user_data.pop('range_msg_id')
        )
        context.user_data['range_start'] = result
        await q.message.reply_text(f'Начало диапазона: {result}')
        cal2, _ = cal_obj.build()
        msg = await q.message.reply_text(
            'Теперь выберите конец диапазона:',
            reply_markup=cal2
        )
        context.user_data['range_msg_id'] = msg.message_id
        return RANGE_END

    # Проверка порядка
    start_ts = pd.to_datetime(context.user_data['range_start'])
    end_ts = pd.to_datetime(result)
    if end_ts < start_ts:
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=context.user_data.pop('range_msg_id')
        )
        cal2, _ = cal_obj.build()
        msg = await q.message.reply_text(
            '❗ Конец раньше начала! Попробуйте снова.',
            reply_markup=cal2
        )
        context.user_data['range_msg_id'] = msg.message_id
        return RANGE_END

    # Всё ок
    await context.bot.delete_message(
        chat_id=q.message.chat.id,
        message_id=context.user_data.pop('range_msg_id')
    )
    context.user_data['range_end'] = result
    await q.message.reply_text(f'Конец диапазона: {result}')

    df_pred = context.user_data['df_pred']
    df_slice = df_pred[(df_pred.index >= start_ts) & (df_pred.index <= end_ts)]

    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_slice.index, df_slice['Forecast'], '-', label='Прогноз')
    ax.set_ylim(0, 1200)
    ax.set_title(f'Прогноз с {start_ts.date()} по {end_ts.date()}')
    ax.legend(); ax.grid(True)
    fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    await q.message.reply_photo(photo=buf)

    keyboard = [
        [InlineKeyboardButton("Повторить прогноз", callback_data="repeat")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")],
        [InlineKeyboardButton("Оставить отзыв", callback_data="feedback")],
        [InlineKeyboardButton("Выход", callback_data="exit")]
    ]
    await q.message.reply_text('Что дальше?', reply_markup=InlineKeyboardMarkup(keyboard))
    return AFTER_FORECAST

async def shock_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    # Удаляем сообщение с вопросом про шок
    await q.message.delete()

    # Сохраняем выбор пользователя
    shock_val = 1.0 if q.data == 'shock_yes' else 0.0
    context.user_data['crises'][-1]['shock'] = shock_val

    # Эхо-сообщение о шоке
    choice_text = 'Да' if shock_val == 1.0 else 'Нет'
    await q.message.reply_text(f'Шоковый кризис: *{choice_text}*', parse_mode='Markdown')

    # Спрашиваем интенсивность и сохраняем его message_id
    int_msg = await q.message.reply_text(
        'Введите интенсивность кризиса (Интенсивность в процентах (0–100) — отражает силу и длительность воздействия.):'
    )
    context.user_data['intensity_msg_id'] = int_msg.message_id

    return INPUT_INTENSITY


async def input_intensity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    text = msg.text.strip().rstrip('%')
    try:
        val = float(text)
        assert 0.0 <= val <= 100.0
    except:
        await msg.reply_text('Введите число от 0 до 100.')
        return INPUT_INTENSITY

    # Удаляем:
    # 1) сообщение с prompt'ом про интенсивность
    # 2) само сообщение пользователя с числом
    chat_id = msg.chat.id
    if 'intensity_msg_id' in context.user_data:
        await context.bot.delete_message(chat_id=chat_id,
                                         message_id=context.user_data.pop('intensity_msg_id'))
    await msg.delete()

    # Сохраняем интенсивность и эхо-сообщение
    context.user_data['crises'][-1]['intensity'] = val / 100.0
    await context.bot.send_message(
        chat_id=chat_id,
        text=f'Интенсивность: *{int(val)}%*',
        parse_mode='Markdown'
    )

    # Переходим дальше
    cur = context.user_data.get('current', 1)
    total = context.user_data.get('num_crises', 0)
    if cur < total:
        context.user_data['current'] = cur + 1
        kb = [[InlineKeyboardButton(label, callback_data=key)]
              for key,label in CRISIS_TYPES.items()]
        await context.bot.send_message(
            chat_id=chat_id,
            text=f'Выберите тип кризиса #{cur+1}:',
            reply_markup=InlineKeyboardMarkup(kb)
        )
        return SELECT_CRISIS_TYPE

    return await launch_forecast(update, context)


async def launch_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await run_forecast(update, context)

async def run_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.callback_query.message if update.callback_query else update.message
    loading = await msg.reply_text("⏳ Генерация прогноза, пожалуйста, подождите...")
    df_hist = pd.read_csv('data/ML_with_crisis.csv', parse_dates=['Date'])
    last_oil = df_hist['Oil_Price'].iloc[-1]
    dates = pd.date_range('2025-01-01', '2030-12-01', freq='MS')
    # build exogenous crises series
    ints = pd.Series(0.0, index=dates)
    shocks = pd.Series(0.0, index=dates)
    for c in context.user_data.get('crises', []):
        s, e = pd.to_datetime(c['start']), pd.to_datetime(c['end'])
        mask = (dates >= s) & (dates <= e)
        ints.loc[mask] = c.get('intensity', 0)
        shocks.loc[mask] = c.get('shock', 0)
    df_exog = pd.DataFrame({
        'Date': dates,
        'Oil_Price': last_oil,
        'crisis_intensity': ints.values,
        'crisis_shock': shocks.values,
        'has_crisis': (ints.values > 0).astype(float),
    })
    for ct in CRISIS_TYPES:
        df_exog[f'crisis_type_{ct}'] = 0.0
    for c in context.user_data.get('crises', []):
        if c['type']:
            mask = (df_exog['Date'] >= pd.to_datetime(c['start'])) & \
                   (df_exog['Date'] <= pd.to_datetime(c['end']))
            df_exog.loc[mask, f'crisis_type_{c["type"]}'] = 1.0
    key, model_func = context.user_data['model_key'], MODEL_OPTIONS[context.user_data['model_key']][1]
    # existing model branches unchanged...
    if key == 'prophet':
        _, prophet_df = model_func(df_exog)
        df_pred = (prophet_df.rename(columns={'yhat_exp': 'Forecast'})
                   .set_index('Date')[['Forecast']])
    elif key == 'tcn':
        df_forecast, _ = model_func(df_exog)
        if 'Date' in df_forecast.columns:
            df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
            df_pred = df_forecast.set_index('Date')[['Forecast']]
        else:
            df_pred = df_forecast[['Forecast']]
    else:
        df_train_full = pd.read_csv('data/ML.csv', parse_dates=['Date'])
        last_hist = df_train_full['Date'].max()
        df_new = df_exog[df_exog['Date'] > last_hist].copy()
        _, y_forecast = model_func(df_new)
        df_raw = pd.DataFrame({'Date': pd.to_datetime(y_forecast.index), 'Forecast': y_forecast.values})
        df_pred = df_raw.set_index('Date')[['Forecast']]
    context.user_data['df_pred'] = df_pred
    await context.bot.delete_message(chat_id=loading.chat.id, message_id=loading.message_id)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_hist['Date'], df_hist['Freight_Price'], label='История 2003–2025')
    ax.plot(df_pred.index, df_pred['Forecast'], '-', label='Прогноз 2025–2030')
    ax.set_ylim(0, 2200)
    ax.set_title('История и прогноз')
    ax.legend(); ax.grid(True)
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    await msg.reply_photo(photo=buf)
    keyboard = [
        [InlineKeyboardButton("Повторить прогноз", callback_data="repeat")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")],
        [InlineKeyboardButton("Оставить отзыв", callback_data="feedback")],
        [InlineKeyboardButton("Выход", callback_data="exit")]
    ]
    await msg.reply_text('Что дальше?', reply_markup=InlineKeyboardMarkup(keyboard))
    return AFTER_FORECAST

async def range_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    cal, _ = DetailedTelegramCalendar(min_date=date(2025, 1, 1), max_date=date(2030, 12, 31)).build()
    msg = await q.message.reply_text('Выберите месяц и год начала диапазона прогноза:', reply_markup=cal)
    context.user_data['range_msg_id'] = msg.message_id
    return RANGE_START

async def calendar_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    cal_obj = DetailedTelegramCalendar(min_date=date(2025, 1, 1), max_date=date(2030, 12, 31))
    result, cal, step = cal_obj.process(q.data)
    if result is None and cal:
        await q.edit_message_text(text=step, reply_markup=cal)
        return CALENDAR_END
    if result is None and cal is None:
        await context.bot.delete_message(chat_id=q.message.chat.id, message_id=q.message.message_id)
        cal2, _ = cal_obj.build()
        await q.message.reply_text('Неверный ввод даты окончания. Выберите снова:', reply_markup=cal2)
        return CALENDAR_END

    # Сбрасываем день и проверяем порядок
    result = result.replace(day=1)
    start_ts = pd.to_datetime(context.user_data['crises'][-1]['start'])
    end_ts   = pd.to_datetime(result)
    if end_ts < start_ts:
        await context.bot.delete_message(chat_id=q.message.chat.id, message_id=q.message.message_id)
        cal2, _ = cal_obj.build()
        await q.message.reply_text('Дата окончания раньше начала! Выберите снова:', reply_markup=cal2)
        return CALENDAR_END

    # Фиксируем окончание
    await q.edit_message_text(f'Дата окончания: {result.strftime("%Y-%m")}')
    context.user_data['crises'][-1]['end'] = result.isoformat()

    # Переход к выбору шока
    kb = [
        [InlineKeyboardButton('Да', callback_data='shock_yes'),
         InlineKeyboardButton('Нет', callback_data='shock_no')]
    ]
    await q.message.reply_text('Шоковый кризис? Нажмите кнопку:', reply_markup=InlineKeyboardMarkup(kb))
    return INPUT_SHOCK

async def range_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    cal_obj = DetailedTelegramCalendar(min_date=date(2025, 1, 1), max_date=date(2030, 12, 31))
    result, cal, step = cal_obj.process(q.data)
    if result is None and cal:
        await q.edit_message_text(text=step, reply_markup=cal)
        return RANGE_END
    if result is None and cal is None:
        if 'range_msg_id' in context.user_data:
            await context.bot.delete_message(chat_id=q.message.chat.id, message_id=context.user_data.pop('range_msg_id'))
        cal2, _ = cal_obj.build()
        msg = await q.message.reply_text('❗ Неверный ввод. Попробуйте снова.', reply_markup=cal2)
        context.user_data['range_msg_id'] = msg.message_id
        return RANGE_END

    # Если первый выбор — начало диапазона
    if 'range_start' not in context.user_data:
        if 'range_msg_id' in context.user_data:
            await context.bot.delete_message(chat_id=q.message.chat.id, message_id=context.user_data.pop('range_msg_id'))
        result = result.replace(day=1)
        context.user_data['range_start'] = result
        await q.message.reply_text(f'Начало диапазона: {result.strftime("%Y-%m")}')
        cal2, _ = cal_obj.build()
        msg = await q.message.reply_text('Теперь выберите месяц и год конца диапазона:', reply_markup=cal2)
        context.user_data['range_msg_id'] = msg.message_id
        return RANGE_END

    # Фиксируем конец диапазона
    result = result.replace(day=1)
    start_ts = pd.to_datetime(context.user_data['range_start'])
    end_ts   = pd.to_datetime(result)
    if end_ts < start_ts:
        if 'range_msg_id' in context.user_data:
            await context.bot.delete_message(chat_id=q.message.chat.id, message_id=context.user_data.pop('range_msg_id'))
        cal2, _ = cal_obj.build()
        msg = await q.message.reply_text('❗ Конец раньше начала! Попробуйте снова.', reply_markup=cal2)
        context.user_data['range_msg_id'] = msg.message_id
        return RANGE_END

    if 'range_msg_id' in context.user_data:
        await context.bot.delete_message(chat_id=q.message.chat.id, message_id=context.user_data.pop('range_msg_id'))
    context.user_data['range_end'] = result
    await q.message.reply_text(f'Конец диапазона: {result.strftime("%Y-%m")}')
    # Отрисовка усечённого прогноза
    df_pred = context.user_data['df_pred']
    df_slice = df_pred[(df_pred.index >= start_ts) & (df_pred.index <= end_ts)]
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_slice.index, df_slice['Forecast'], '-', label='Прогноз')
    ax.set_ylim(0, 1200)
    ax.set_title(f'Прогноз с {start_ts.date()} по {end_ts.date()}')
    ax.legend(); ax.grid(True)
    fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    await q.message.reply_photo(photo=buf)

    # Кнопки дальнейших действий
    keyboard = [
        [InlineKeyboardButton("Повторить прогноз", callback_data="repeat")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")],
        [InlineKeyboardButton("Оставить отзыв", callback_data="feedback")],
        [InlineKeyboardButton("Выход", callback_data="exit")]
    ]
    await q.message.reply_text('Что дальше?', reply_markup=InlineKeyboardMarkup(keyboard))
    return AFTER_FORECAST

async def feedback_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    await q.message.reply_text(
        "🙏 Пожалуйста, оставьте ваш отзыв:\n"
        "1) Что понравилось?\n"
        "2) Что улучшить?\n"
        "3) Дополнительные комментарии"
    )
    return FEEDBACK


async def feedback_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user = os.getenv("GMAIL_USER")
    pwd = os.getenv("GMAIL_PASS")
    msg = MIMEText(text, _charset="utf-8")
    msg["Subject"] = "Отзыв DemandForecastBot"
    msg["From"] = user
    msg["To"] = user
    with SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(user, pwd)
        smtp.send_message(msg)
    await update.message.reply_text("Спасибо за ваш отзыв! Для перезапуска нажмите /start")
    return ConversationHandler.END


async def repeat_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    context.user_data.clear()
    return await start(update, context)


async def exit_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    await q.message.reply_text(
        'Бот завершает работу. До встречи! Для перезапуска нажмите /start'
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.callback_query.message if update.callback_query else update.message
    await msg.reply_text('Отменено.')
    return ConversationHandler.END


if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    conv = ConversationHandler(
    entry_points=[CommandHandler('start', start)],
    states={
        SELECT_MODEL:              [CallbackQueryHandler(model_chosen)],
        SELECT_OIL_SOURCE:         [CallbackQueryHandler(oil_source_chosen)],
        SELECT_PREDEFINED_CRISES:  [CallbackQueryHandler(predefined_crises_chosen)],
        ASK_NUM_CRISES:            [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_num_crises)],
        SELECT_CRISIS_TYPE:        [CallbackQueryHandler(select_crisis_type)],
        CALENDAR_START:            [CallbackQueryHandler(calendar_start)],
        CALENDAR_END:              [CallbackQueryHandler(calendar_end)],
        INPUT_SHOCK:               [CallbackQueryHandler(shock_chosen, pattern='^shock_')],
        INPUT_INTENSITY:           [MessageHandler(filters.TEXT & ~filters.COMMAND, input_intensity)],
        AFTER_FORECAST:            [
            CallbackQueryHandler(repeat_forecast, pattern='^repeat$'),
            CallbackQueryHandler(range_start,    pattern='^range$'),
            CallbackQueryHandler(feedback_start, pattern='^feedback$'),
            CallbackQueryHandler(exit_bot,       pattern='^exit$'),
        ],
        RANGE_START:               [CallbackQueryHandler(range_end)],
        RANGE_END:                 [CallbackQueryHandler(range_end)],
        FEEDBACK:                  [MessageHandler(filters.TEXT & ~filters.COMMAND, feedback_received)],
    },
    fallbacks=[CommandHandler('cancel', cancel)],
    allow_reentry=True
)
    app.add_handler(conv)
    app.run_polling()

