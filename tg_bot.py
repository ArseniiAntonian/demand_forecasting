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

USERS_FILE = 'users.txt'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_user(user_id):
    with open(USERS_FILE, 'a') as f:
        f.write(f"{user_id}\n")
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
    FEEDBACK,
    SELECT_START_YEAR, SELECT_START_MONTH, SELECT_END_YEAR, SELECT_END_MONTH,
    SELECT_RANGE_START_YEAR, SELECT_RANGE_START_MONTH, SELECT_RANGE_END_YEAR, SELECT_RANGE_END_MONTH
) = range(21)

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
        SELECT_START_YEAR:  [CallbackQueryHandler(start_year_chosen)],
        SELECT_START_MONTH: [CallbackQueryHandler(start_month_chosen)],
        SELECT_END_YEAR:    [CallbackQueryHandler(end_year_chosen)],
        SELECT_END_MONTH:   [CallbackQueryHandler(end_month_chosen)],
        INPUT_SHOCK:              [CallbackQueryHandler(shock_chosen, pattern='^shock_')],
        INPUT_INTENSITY:           [MessageHandler(filters.TEXT & ~filters.COMMAND, input_intensity)],
        AFTER_FORECAST:            [
            CallbackQueryHandler(repeat_forecast, pattern='^repeat$'),
            CallbackQueryHandler(range_start,    pattern='^range$'),
            CallbackQueryHandler(feedback_start, pattern='^feedback$'),
            CallbackQueryHandler(exit_bot,       pattern='^exit$'),
        ],
        SELECT_RANGE_START_YEAR: [CallbackQueryHandler(range_start_year_chosen)],
        SELECT_RANGE_START_MONTH:[CallbackQueryHandler(range_start_month_chosen)],
        SELECT_RANGE_END_YEAR:   [CallbackQueryHandler(range_end_year_chosen)],
        SELECT_RANGE_END_MONTH:  [CallbackQueryHandler(range_end_month_chosen)],
        FEEDBACK:                  [MessageHandler(filters.TEXT & ~filters.COMMAND, feedback_received)],
    }


# Entry point
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message or update.callback_query.message
    user_id = str(update.effective_user.id)
    users = load_users()
    is_new_user = user_id not in users
    if is_new_user:
        save_user(user_id)
        users.add(user_id)
    if update.callback_query:
        await update.callback_query.answer()

    welcome = (
        "Добро пожаловать в бот-прогнозист фрахтовых цен!\n\n"
        "Что умеет бот:\n"
        "  • Строит график фактических цен (2003–2025) и прогноз (2025–2030).\n"
        "  • Поддерживает три модели:\n" 
        "           1) Аддитивная модель временных рядов\n"
        "           2) Нейросетевая модель\n"
        "           3) Ансамблевый метод машинного обучения\n"
        "  • Учтёт ваши сценарии кризисов: тип, даты и интенсивность.\n\n"
        "Как начать работу:\n"
        "1️⃣ Шаг 1: Нажмите на кнопку с моделью, которую хотите использовать.\n"
        "2️⃣ Шаг 2: Выберите источник прогноза нефти — Brent или WTI.\n"
        "   • Brent — эталон для Европы и Азии, чувствителен к геополитике.\n"
        "   • WTI — ориентир для США, более волатилен и зависит от внутренней добычи.\n"
        "3️⃣ Шаг 3: Укажите кризисы, которые следует учесть:\n"
        "   • Вы можете:\n"
        "     – Выбрать из готового списка (до 7 кризисов)\n"
        "     – И/или ввести собственные сценарии вручную\n"
        "   • Для каждого кризиса указываются:\n"
        "     – Тип (финансовый, геополитический и т. д.)\n"
        "     – Дата начала и окончания\n"
        "     – Интенсивность (0–100%) — отражает силу и длительность воздействия\n"
        "     – Шоковость (да/нет) — указывает, произошло ли событие резко или развивалось постепенно.\n\n"
        "4️⃣ Шаг 4: Дождитесь построения графика. После будут четыре кнопки:\n"
        "   • «Повторить прогноз» — начать всё сначала.\n"
        "   • «Выбрать интервал прогноза» — посмотреть прогноз за выбранный период.\n"
        "   • «Оставить отзыв» — поделиться впечатлениями.\n"
        "   • «Выход» — завершить работу бота.\n\n"
        "После прогноза вы сможете вернуться и снова выбрать модель, указать новые кризисы или просто выйти.\n\n"
        "*P.S. Если бот не отвечает, перезапустите его через команду /start*"
    )

    await msg.reply_text(welcome, parse_mode="Markdown")

    kb = [[InlineKeyboardButton(name, callback_data=key)]
          for key, (name, _) in MODEL_OPTIONS.items()]
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
    lines = ['*Выберите от 1 до 7 кризисов, которые будут учитываться в прогнозе (Можно выбрать один, несколько или все), также вы можете задать параметры кризисов вручную, нажав на кнопку "Ввести вручную".*\n\n'
    '*Список кризисов:*']
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
    q = update.callback_query
    await q.answer()
    d = q.data

    if d == 'no_crises':
        context.user_data['crises'] = []
        context.user_data['num_crises'] = 0
        return await launch_forecast(update, context)

    if d == 'manual_crises':
        # Добавляем уже выбранные предопределённые кризисы (если есть)
        sel = context.user_data.get('pre_selected', [])
        if sel:
            context.user_data.setdefault('crises', [])
            for i in sel:
                context.user_data['crises'].append({
                    'type': PREDEFINED_CRISES[i-1]['type'],
                    'start': PREDEFINED_CRISES[i-1]['start'],
                    'end':   PREDEFINED_CRISES[i-1]['end'],
                    'intensity': PREDEFINED_CRISES[i-1]['intensity'],
                    'shock':     PREDEFINED_CRISES[i-1]['shock']
                })
        # Переходим к ручному вводу количества кризисов
        await q.message.reply_text('Сколько дополнительных кризисов (0–10)?')
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

    # Считаем уже выбранные (предопределённые) кризисы
    existing = len(context.user_data.get('crises', []))
    context.user_data['num_crises'] = existing + n
    context.user_data.setdefault('crises', [])

    # Если ничего не нужно добавлять — запускаем прогноз
    if n == 0:
        return await launch_forecast(update, context)

    # Начинаем нумерацию с корректного номера
    context.user_data['current'] = existing + 1

    # Строим клавиатуру типов кризисов
    kb = [[InlineKeyboardButton(label, callback_data=key)]
          for key, label in CRISIS_TYPES.items()]
    await update.message.reply_text(
        f'Выберите тип кризиса #{context.user_data["current"]}:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_CRISIS_TYPE


async def select_crisis_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    await q.message.delete()

    ctype_key = q.data
    idx = context.user_data['current']
    context.user_data['crises'].append({'type': ctype_key})
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Тип кризиса #{idx}: *{CRISIS_TYPES[ctype_key]}*',
        parse_mode='Markdown'
    )

    # теперь вместо DetailedTelegramCalendar запускаем выбор года
    years = list(range(2025, 2031))
    kb = [[InlineKeyboardButton(str(y), callback_data=str(y))] for y in years]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Выберите год начала кризиса #{idx}:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_START_YEAR
async def start_year_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    year = int(q.data)
    context.user_data['crises'][-1]['start_year'] = year
    await q.message.delete()

    MONTHS = [
        ("Январь","01"),("Февраль","02"),("Март","03"),("Апрель","04"),
        ("Май","05"),("Июнь","06"),("Июль","07"),("Август","08"),
        ("Сентябрь","09"),("Октябрь","10"),("Ноябрь","11"),("Декабрь","12")
    ]
    kb = [[InlineKeyboardButton(name, callback_data=code)] for name,code in MONTHS]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Выберите месяц начала:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_START_MONTH

# === Новый хендлер: обработка выбранного месяца начала
async def start_month_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    month = int(q.data)
    year = context.user_data['crises'][-1]['start_year']
    start_dt = date(year, month, 1)
    context.user_data['crises'][-1]['start'] = start_dt.isoformat()
    await q.message.delete()
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Дата начала: *{start_dt.strftime("%Y-%m")}*',
        parse_mode='Markdown'
    )

    years = list(range(year, 2031))
    kb = [[InlineKeyboardButton(str(y), callback_data=str(y))] for y in years]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Выберите год окончания:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_END_YEAR

# === Новый хендлер: обработка выбранного года окончания
async def end_year_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    end_year = int(q.data)
    context.user_data['crises'][-1]['end_year'] = end_year
    await q.message.delete()

    MONTHS = [
        ("Январь","01"),("Февраль","02"),("Март","03"),("Апрель","04"),
        ("Май","05"),("Июнь","06"),("Июль","07"),("Август","08"),
        ("Сентябрь","09"),("Октябрь","10"),("Ноябрь","11"),("Декабрь","12")
    ]
    kb = [[InlineKeyboardButton(name, callback_data=code)] for name,code in MONTHS]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Выберите месяц окончания:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_END_MONTH

# === Новый хендлер: обработка выбранного месяца окончания
async def end_month_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    month = int(q.data)
    year = context.user_data['crises'][-1]['end_year']
    end_dt = date(year, month, 1)

    start_ts = pd.to_datetime(context.user_data['crises'][-1]['start'])
    if pd.Timestamp(end_dt) < start_ts:
        await q.message.delete()
        # заново предлагаем выбрать год окончания, начиная с года начала
        start_year = context.user_data['crises'][-1]['start_year']
        years = list(range(start_year, 2031))
        kb = [[InlineKeyboardButton(str(y), callback_data=str(y))] for y in years]
        await context.bot.send_message(
            chat_id=q.message.chat.id,
            text='❗ Дата окончания раньше начала. Выберите год окончания заново.',
            reply_markup=InlineKeyboardMarkup(kb)
        )
        return SELECT_END_YEAR

    context.user_data['crises'][-1]['end'] = end_dt.isoformat()
    await q.message.delete()
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Дата окончания: *{end_dt.strftime("%Y-%m")}*',
        parse_mode='Markdown'
    )

    kb = [
        [InlineKeyboardButton('Да', callback_data='shock_yes'),
         InlineKeyboardButton('Нет', callback_data='shock_no')]
    ]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Шоковый кризис?(Шоковость (да/нет) — указывает, произошло ли событие резко или развивалось постепенно.)',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return INPUT_SHOCK
    
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
    """
    Запускает прогноз, выводит график истории и прогноза с шапкой,
    содержащей выбор пользователя (модель, нефть, список кризисов),
    и подписывает ось Y как USD.
    """
    # Определяем, куда отправлять сообщения
    msg = update.callback_query.message if update.callback_query else update.message
    # Показываем индикатор загрузки
    loading = await msg.reply_text("⏳ Генерация прогноза, пожалуйста, подождите...")

    # --- Загрузка и подготовка данных ---
    df_hist = pd.read_csv('data/ML_with_crisis.csv', parse_dates=['Date'])
    last_oil = df_hist['Oil_Price'].iloc[-1]
    dates = pd.date_range('2025-01-01', '2030-12-01', freq='MS')

    # Строим exogenous dataframe с кризисами
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
    # dummy-переменные по типам кризисов
    for ct in CRISIS_TYPES:
        df_exog[f'crisis_type_{ct}'] = 0.0
    for c in context.user_data.get('crises', []):
        if c['type']:
            mask = (df_exog['Date'] >= pd.to_datetime(c['start'])) & \
                   (df_exog['Date'] <= pd.to_datetime(c['end']))
            df_exog.loc[mask, f'crisis_type_{c["type"]}'] = 1.0

    # Убираем сообщение "жду"
    # await context.bot.delete_message(chat_id=loading.chat.id, message_id=loading.message_id)

    # Запускаем выбранную модель
    key, model_func = context.user_data['model_key'], MODEL_OPTIONS[context.user_data['model_key']][1]
    if key == 'prophet':
        _, prophet_df = model_func(df_exog)
        df_pred = (prophet_df
                   .rename(columns={'yhat_exp': 'Forecast'})
                   .set_index('Date')[['Forecast']])
    elif key == 'tcn':
        df_forecast, _ = model_func(df_exog)
        if 'Date' in df_forecast.columns:
            df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
            df_pred = df_forecast.set_index('Date')[['Forecast']]
        else:
            df_pred = df_forecast[['Forecast']]
    else:  # LightGBM
        df_train_full = pd.read_csv('data/ML.csv', parse_dates=['Date'])
        last_hist = df_train_full['Date'].max()
        df_new = df_exog[df_exog['Date'] > last_hist].copy()
        _, y_forecast = model_func(df_new)
        df_raw = pd.DataFrame({
            'Date': pd.to_datetime(y_forecast.index),
            'Forecast': y_forecast.values
        })
        df_pred = df_raw.set_index('Date')[['Forecast']]

    # Сохраняем прогноз в user_data для дальнейших действий
    context.user_data['df_pred'] = df_pred

    # --- Заголовок с выбором пользователя ---
    model_name = MODEL_OPTIONS[key][0]
    oil        = context.user_data['oil_source'].upper()
    crises     = context.user_data.get('crises', [])

    if crises:
        lines = []
        for i, c in enumerate(crises, start=1):
            lines.append(
                f"{i}) {CRISIS_TYPES[c['type']]} "
                f"{c['start'][:7]}–{c['end'][:7]}, "
                f"шок:{'да' if c['shock'] else 'нет'}, "
                f"инт.:{int(c['intensity']*100)}%"
            )
        crises_block = "\n".join(lines)
    else:
        crises_block = "Без кризисов"

    header = (
        f"Модель: {model_name}; Нефть: {oil}\n"
        f"Кризисы:\n{crises_block}"
    )

    # --- Рисуем график ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_hist['Date'], df_hist['Freight_Price'],
            label='История 2003–2025')
    ax.plot(df_pred.index, df_pred['Forecast'],
            '-', label='Прогноз 2025–2030', color='orange')
    ax.set_ylim(0, max(df_hist['Freight_Price'].max(), df_pred['Forecast'].max()) * 1.1)
    ax.set_ylabel('USD')
    ax.set_title(header, loc='left')
    ax.legend()
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    await context.bot.delete_message(chat_id=loading.chat.id, message_id=loading.message_id)
    # Отправляем картинку
    await msg.reply_photo(photo=buf)

    # Финальное меню действий
    keyboard = [
        [InlineKeyboardButton("Повторить прогноз", callback_data="repeat")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")],
        [InlineKeyboardButton("Оставить отзыв", callback_data="feedback")],
        [InlineKeyboardButton("Выход", callback_data="exit")],
    ]
    await msg.reply_text('Что дальше?', reply_markup=InlineKeyboardMarkup(keyboard))
    return AFTER_FORECAST


async def range_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    await q.message.delete()
    years = list(range(2025, 2031))
    kb = [[InlineKeyboardButton(str(y), callback_data=str(y))] for y in years]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Выберите год начала диапазона:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_RANGE_START_YEAR

async def range_start_year_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    start_year = int(q.data)
    context.user_data['range_start_year'] = start_year
    await q.message.delete()

    MONTHS = [
        ("Январь","01"),("Февраль","02"),("Март","03"),("Апрель","04"),
        ("Май","05"),("Июнь","06"),("Июль","07"),("Август","08"),
        ("Сентябрь","09"),("Октябрь","10"),("Ноябрь","11"),("Декабрь","12")
    ]
    kb = [[InlineKeyboardButton(name, callback_data=code)] for name,code in MONTHS]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Выберите месяц начала диапазона:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_RANGE_START_MONTH

# 4) Выбор месяца начала → запрос года конца
async def range_start_month_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    m = int(q.data)
    y = context.user_data['range_start_year']
    context.user_data['range_start'] = date(y, m, 1).isoformat()
    await q.message.delete()
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Начало диапазона: *{y}-{m:02d}*',
        parse_mode='Markdown'
    )

    years = list(range(y, 2031))
    kb = [[InlineKeyboardButton(str(year), callback_data=str(year))] for year in years]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Выберите год конца диапазона:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_RANGE_END_YEAR

# 5) Выбор года конца → запрос месяца конца
async def range_end_year_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    end_year = int(q.data)
    context.user_data['range_end_year'] = end_year
    await q.message.delete()

    MONTHS = [
        ("Январь","01"),("Февраль","02"),("Март","03"),("Апрель","04"),
        ("Май","05"),("Июнь","06"),("Июль","07"),("Август","08"),
        ("Сентябрь","09"),("Октябрь","10"),("Ноябрь","11"),("Декабрь","12")
    ]
    kb = [[InlineKeyboardButton(name, callback_data=code)] for name,code in MONTHS]
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text='Выберите месяц конца диапазона:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_RANGE_END_MONTH

# 6) Выбор месяца конца → финализируем и рисуем график
async def range_end_month_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    month = int(q.data)
    year  = context.user_data['range_end_year']
    end_dt = date(year, month, 1)

    # Проверка порядка
    start_ts = pd.to_datetime(context.user_data['range_start'])
    if pd.Timestamp(end_dt) < start_ts:
        await q.message.delete()
        await context.bot.send_message(
            chat_id=q.message.chat.id,
            text='❗ Конец раньше начала! Начните выбор конца года заново.'
        )
        return SELECT_RANGE_END_YEAR

    context.user_data['range_end'] = end_dt.isoformat()
    await q.message.delete()
    await context.bot.send_message(
        chat_id=q.message.chat.id,
        text=f'Конец диапазона: *{year}-{month:02d}*',
        parse_mode='Markdown'
    )

    # Усечённый прогноз
    df_pred = context.user_data['df_pred']
    rs = pd.to_datetime(context.user_data['range_start'])
    re = pd.to_datetime(context.user_data['range_end'])
    df_slice = df_pred[(df_pred.index >= rs) & (df_pred.index <= re)]

    # Заголовок (тот же формат, что в run_forecast)
    key        = context.user_data['model_key']
    model_name = MODEL_OPTIONS[key][0]
    oil        = context.user_data['oil_source'].upper()
    crises     = context.user_data.get('crises', [])
    if crises:
        items = [
            f"{i}) {CRISIS_TYPES[c['type']]} {c['start'][:7]}–{c['end'][:7]}, "
            f"шок:{'да' if c['shock'] else 'нет'}, инт.:{int(c['intensity']*100)}%"
            for i, c in enumerate(crises, 1)
        ]
        crises_block = "\n".join(items)
    else:
        crises_block = "Без кризисов"
    header = (
        f"Модель: {model_name}; Нефть: {oil}\n"
        f"Кризисы:\n{crises_block}"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_slice.index, df_slice['Forecast'], '-', label='Прогноз', color='orange')
    ax.set_ylim(0, 1700)
    ax.set_ylabel('USD')
    ax.set_title(header, loc='left')
    ax.legend()
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    await q.message.reply_photo(photo=buf)

    keyboard = [
        [InlineKeyboardButton("Повторить прогноз", callback_data="repeat")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")],
        [InlineKeyboardButton("Оставить отзыв", callback_data="feedback")],
        [InlineKeyboardButton("Выход", callback_data="exit")],
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
        SELECT_START_YEAR:  [CallbackQueryHandler(start_year_chosen)],
        SELECT_START_MONTH: [CallbackQueryHandler(start_month_chosen)],
        SELECT_END_YEAR:    [CallbackQueryHandler(end_year_chosen)],
        SELECT_END_MONTH:   [CallbackQueryHandler(end_month_chosen)],
        INPUT_SHOCK:               [CallbackQueryHandler(shock_chosen, pattern='^shock_')],
        INPUT_INTENSITY:           [MessageHandler(filters.TEXT & ~filters.COMMAND, input_intensity)],
        AFTER_FORECAST:            [
            CallbackQueryHandler(repeat_forecast, pattern='^repeat$'),
            CallbackQueryHandler(range_start,    pattern='^range$'),
            CallbackQueryHandler(feedback_start, pattern='^feedback$'),
            CallbackQueryHandler(exit_bot,       pattern='^exit$'),
        ],
        RANGE_START:             [CallbackQueryHandler(range_start)],
        SELECT_RANGE_START_YEAR: [CallbackQueryHandler(range_start_year_chosen)],
        SELECT_RANGE_START_MONTH:[CallbackQueryHandler(range_start_month_chosen)],
        SELECT_RANGE_END_YEAR:   [CallbackQueryHandler(range_end_year_chosen)],
        SELECT_RANGE_END_MONTH:  [CallbackQueryHandler(range_end_month_chosen)],
        FEEDBACK:                  [MessageHandler(filters.TEXT & ~filters.COMMAND, feedback_received)],
    },
    fallbacks=[CommandHandler('cancel', cancel)],
    allow_reentry=True
)
    app.add_handler(conv)
    app.run_polling()

