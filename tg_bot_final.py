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
    ASK_NUM_CRISES,
    SELECT_CRISIS_TYPE,
    CALENDAR_START,
    CALENDAR_END,
    INPUT_INTENSITY,
    AFTER_FORECAST,
    RANGE_START,
    RANGE_END,
    FEEDBACK
) = range(11)

MODEL_OPTIONS = {
    'prophet': ('Prophet',    forecast_prophet),
    'tcn':     ('TCN',        forecast_last_data_w_exogs),
    'gb':      ('LightGBM',   forecast_lgb),
}

CRISIS_TYPES = {
    'Financial':   'Финансовый',
    'Pandemic':    'Пандемический',
    'Geopolitical':'Геополитический',
    'Natural':     'Природный',
    'Logistical':  'Логистический',
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # выбираем, куда слать: update.message или update.callback_query.message
    msg = update.message or update.callback_query.message
    # отвечаем на callback (если он есть), чтобы убрать “часики”
    if update.callback_query:
        await update.callback_query.answer()

    welcome_text = (
        "Добро пожаловать в бот-прогнозист фрахтовых цен!\n\n"
        "Что умеет бот:\n"
        "  • Строит график фактических цен (2003–2025) и прогноз (2025–2030).\n"
        "  • Поддерживает три модели: Prophet, TCN и LightGBM.\n"
        "  • Учтёт ваши сценарии кризисов: тип, даты и интенсивность.\n\n"
        "Как начать работу:\n"
        "1️⃣ Шаг 1: Нажмите на кнопку с моделью, которую хотите использовать.\n"
        "2️⃣ Шаг 2: Выберите источник прогноза нефти — Brent или WTI.\n"
        "3️⃣ Шаг 3: Укажите, сколько кризисов заложить (0–10).\n"
        "   – Для каждого кризиса выберите:\n"
        "     • Тип кризиса (финансовый, пандемический и т. д.).\n"
        "     • Дату начала и окончания через календарь.\n"
        "     • Интенсивность в процентах (0–100%).\n"
        "4️⃣ Шаг 4: Дождитесь построения графика. После будут четыре кнопки:\n"
        "   • «Повторить прогноз» — начать всё сначала.\n"
        "   • «Выбрать интервал прогноза» — посмотреть прогноз за выбранный период.\n"
        "   • «Оставить отзыв» — поделиться впечатлениями.\n"
        "   • «Выход» — завершить работу бота.\n\n"
        "После прогноза вы сможете вернуться и снова выбрать модель, указать новые кризисы или просто выйти."
    )

    await msg.reply_text(welcome_text)

    keyboard = [
        [InlineKeyboardButton(name, callback_data=key)]
        for key, (name, _) in MODEL_OPTIONS.items()
    ]
    await msg.reply_text(
        '⏩ Пожалуйста, выберите модель прогнозирования:',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SELECT_MODEL


async def model_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    key = q.data; context.user_data['model_key'] = key
    name, _ = MODEL_OPTIONS[key]
    await q.message.reply_text(f'Модель: {name}')
    kb = [[InlineKeyboardButton('Brent', callback_data='brent'),
           InlineKeyboardButton('WTI',   callback_data='wti')]]
    await q.message.reply_text(
        'Выберите источник нефти:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_OIL_SOURCE


async def oil_source_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    context.user_data['oil_source'] = q.data
    await q.message.reply_text(f'Нефть: {q.data}')
    await q.message.reply_text('Сколько кризисов задать? (0–10)')
    return ASK_NUM_CRISES


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
    q = update.callback_query; await q.answer()
    ctype_key = q.data
    idx = context.user_data['current']
    context.user_data['crises'].append({'type': ctype_key})
    cal, _ = DetailedTelegramCalendar(
        min_date=date(2025,1,1),
        max_date=date(2030,12,31)
    ).build()
    await q.message.reply_text(
        f'Выберите дату начала для кризиса #{idx} ({CRISIS_TYPES[ctype_key]}):',
        reply_markup=cal
    )
    return CALENDAR_START


async def calendar_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    cal_obj = DetailedTelegramCalendar(
        min_date=date(2025,1,1), max_date=date(2030,12,31)
    )
    result, cal, step = cal_obj.process(q.data)

    # ещё листаем календарь
    if result is None and cal:
        await q.edit_message_text(text=step, reply_markup=cal)
        return CALENDAR_START

    # неверный ввод не из календаря
    if result is None and cal is None:
        # удаляем сообщение-панель целиком
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=q.message.message_id
        )
        # пишем об ошибке
        err = await q.message.reply_text(
            'Неверный ввод даты начала. Пожалуйста, выберите заново.'
        )
        # показываем заново
        cal2, _ = cal_obj.build()
        await q.message.reply_text(
            'Выберите дату начала:',
            reply_markup=cal2
        )
        return CALENDAR_START

    # всё ок, удаляем предыдущую ошибку (если была)
    if 'start_error_id' in context.user_data:
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=context.user_data.pop('start_error_id')
        )

    # сохраняем результат
    await q.edit_message_text(f'Дата начала: {result}')
    context.user_data['crises'][-1]['start'] = result.isoformat()

    # запускаем выбор конца
    cal2, _ = cal_obj.build()
    await q.message.reply_text(
        'Выберите дату окончания:',
        reply_markup=cal2
    )
    return CALENDAR_END



async def calendar_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    cal_obj = DetailedTelegramCalendar(
        min_date=date(2025,1,1), max_date=date(2030,12,31)
    )
    result, cal, step = cal_obj.process(q.data)

    # ещё листаем календарь
    if result is None and cal:
        await q.edit_message_text(text=step, reply_markup=cal)
        return CALENDAR_END

    # неверный ввод не из календаря
    if result is None and cal is None:
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=q.message.message_id
        )
        err = await q.message.reply_text(
            'Неверный ввод даты окончания. Пожалуйста, выберите заново.'
        )
        cal2, _ = cal_obj.build()
        await q.message.reply_text(
            'Выберите дату окончания:',
            reply_markup=cal2
        )
        return CALENDAR_END

    # удаляем старую ошибку
    if 'end_error_id' in context.user_data:
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=context.user_data.pop('end_error_id')
        )

    # проверяем порядок
    start_ts = pd.to_datetime(context.user_data['crises'][-1]['start'])
    end_ts   = pd.to_datetime(result)
    if end_ts < start_ts:
        # удаляем панель
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=q.message.message_id
        )
        err2 = await q.message.reply_text(
            'Дата окончания раньше начала! Пожалуйста, выберите снова.'
        )
        cal2, _ = cal_obj.build()
        await q.message.reply_text(
            'Выберите дату окончания:',
            reply_markup=cal2
        )
        return CALENDAR_END

    # сохраняем и движемся дальше
    await q.edit_message_text(f'Дата окончания: {result}')
    context.user_data['crises'][-1]['end'] = result.isoformat()
    await q.message.reply_text('Введите интенсивность в процентах (0–100):')
    return INPUT_INTENSITY



async def input_intensity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text.strip().rstrip('%')
        val = float(text); assert 0.0 <= val <= 100.0
    except:
        await update.message.reply_text('Введите число от 0 до 100.')
        return INPUT_INTENSITY
    context.user_data['crises'][-1]['intensity'] = val / 100.0

    cur = context.user_data['current']
    total = context.user_data['num_crises']
    if cur < total:
        context.user_data['current'] += 1
        kb = [[InlineKeyboardButton(label, callback_data=key)]
              for key, label in CRISIS_TYPES.items()]
        await update.message.reply_text(
            f'Выберите тип кризиса #{cur+1}:',
            reply_markup=InlineKeyboardMarkup(kb)
        )
        return SELECT_CRISIS_TYPE

    return await launch_forecast(update, context)


async def launch_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await run_forecast(update, context)

async def run_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # определяем, куда писать (callback или обычное сообщение)
    msg = update.callback_query.message if update.callback_query else update.message

    # 1️⃣ Показываем индикатор загрузки
    loading = await msg.reply_text("⏳ Генерация прогноза, пожалуйста, подождите...")

    # 2️⃣ Загружаем историю и подготавливаем exog
    df_hist = pd.read_csv('data/ML_with_crisis.csv', parse_dates=['Date'])
    last_oil = df_hist['Oil_Price'].iloc[-1]

    dates = pd.date_range('2025-01-01', '2030-12-01', freq='MS')
    ints = pd.Series(0.0, index=dates)
    for c in context.user_data.get('crises', []):
        s, e = pd.to_datetime(c['start']), pd.to_datetime(c['end'])
        ints.loc[(dates >= s) & (dates <= e)] = c['intensity']

    df_exog = pd.DataFrame({
        'Date': dates,
        'Oil_Price': last_oil,
        'crisis_intensity': ints.values,
        'crisis_shock': 0.0,
    })
    # добавляем булев признак наличия кризиса
    df_exog['has_crisis'] = (df_exog['crisis_intensity'] > 0).astype(float)
    # dummy-переменные по типам кризисов
    for ct in CRISIS_TYPES:
        df_exog[f'crisis_type_{ct}'] = 0.0
    for c in context.user_data.get('crises', []):
        s, e = pd.to_datetime(c['start']), pd.to_datetime(c['end'])
        mask = (df_exog['Date'] >= s) & (df_exog['Date'] <= e)
        df_exog.loc[mask, f'crisis_type_{c["type"]}'] = 1.0

    # 3️⃣ Выбираем модель и делаем прогноз
    key, model_func = context.user_data['model_key'], MODEL_OPTIONS[context.user_data['model_key']][1]

    if key == 'prophet':
        _, prophet_df = model_func(df_exog)
        df_pred = (
            prophet_df
            .rename(columns={'yhat_exp': 'Forecast'})
            .set_index('Date')[['Forecast']]
        )

    elif key == 'tcn':
        # tcn_forecast возвращает (df_forecast, df_hist)
        df_forecast, _ = model_func(df_exog)
        df_forecast = df_forecast.copy()
        if 'Date' in df_forecast.columns:
            df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
            df_pred = df_forecast.set_index('Date')[['Forecast']]
        else:
            df_pred = df_forecast[['Forecast']]

    else:  # LightGBM и другие
        df_train_full = pd.read_csv('data/ML.csv', parse_dates=['Date'])
        last_hist = df_train_full['Date'].max()
        df_new = df_exog[df_exog['Date'] > last_hist].copy()
        _, y_forecast = model_func(df_new)

        # Явно формируем DataFrame с Date и Forecast
        if isinstance(y_forecast, pd.Series):
            df_raw = pd.DataFrame({
                'Date': y_forecast.index,
                'Forecast': y_forecast.values
            })
        else:
            df_raw = y_forecast.reset_index()
            df_raw.columns = ['Date', 'Forecast']
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_pred = df_raw.set_index('Date')[['Forecast']]

    context.user_data['df_pred'] = df_pred

    # 4️⃣ Удаляем «генерация…»
    await context.bot.delete_message(chat_id=loading.chat.id, message_id=loading.message_id)

    # 5️⃣ Строим график истории + прогноза
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_hist['Date'], df_hist['Freight_Price'], label='История 2003–2025')
    ax.plot(df_pred.index, df_pred['Forecast'], '-', label='Прогноз 2025–2030')
    ax.set_ylim(0, 2200)
    ax.set_title('История и прогноз')
    ax.legend(); ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    await msg.reply_photo(photo=buf)

    # 6️⃣ Кнопки, включая повторный выбор диапазона
    keyboard = [
        [InlineKeyboardButton("Повторить прогноз",         callback_data="repeat")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")],
        [InlineKeyboardButton("Оставить отзыв",            callback_data="feedback")],
        [InlineKeyboardButton("Выход",                      callback_data="exit")]
    ]
    await msg.reply_text('Что дальше?', reply_markup=InlineKeyboardMarkup(keyboard))
    return AFTER_FORECAST





# 1) Стартовый хэндлер: сохраняем сообщение-календарь в user_data
async def range_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    cal, _ = DetailedTelegramCalendar(
        min_date=date(2025,1,1),
        max_date=date(2030,12,31)
    ).build()

    msg = await q.message.reply_text(
        'Выберите начало диапазона прогноза:',
        reply_markup=cal
    )
    context.user_data['range_msg_id'] = msg.message_id

    return RANGE_START



async def range_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    cal_obj = DetailedTelegramCalendar(
        min_date=date(2025,1,1),
        max_date=date(2030,12,31)
    )
    result, cal, step = cal_obj.process(q.data)

    # Листаем месяцы
    if result is None and cal:
        await q.edit_message_text(text=step, reply_markup=cal)
        return RANGE_END

    # Неверный ввод (не календарь)
    if result is None and cal is None:
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=context.user_data.pop('range_msg_id')
        )
        await q.message.reply_text('❗ Неверный ввод. Попробуйте снова.')
        cal2, _ = cal_obj.build()
        msg = await q.message.reply_text(
            'Выберите начало диапазона прогноза:',
            reply_markup=cal2
        )
        context.user_data['range_msg_id'] = msg.message_id
        return RANGE_END

    # Фиксируем начало диапазона
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

    # Проверяем, не раньше ли конец чем начало
    start_ts = pd.to_datetime(context.user_data['range_start'])
    end_ts   = pd.to_datetime(result)
    if end_ts < start_ts:
        await context.bot.delete_message(
            chat_id=q.message.chat.id,
            message_id=context.user_data.pop('range_msg_id')
        )
        await q.message.reply_text('❗ Конец раньше начала! Попробуйте снова.')
        cal2, _ = cal_obj.build()
        msg = await q.message.reply_text(
            'Выберите конец диапазона:',
            reply_markup=cal2
        )
        context.user_data['range_msg_id'] = msg.message_id
        return RANGE_END

    # Всё ок — удаляем последний календарь, показываем результат и график
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
    ax.plot(df_slice.index, df_slice['Forecast'], '-', color='orange', label='Прогноз')
    ax.set_ylim(0, 1200)
    ax.set_title(f'Прогноз с {start_ts.date()} по {end_ts.date()}')
    ax.legend(); ax.grid(True)
    fig.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    await q.message.reply_photo(photo=buf)

    # Здесь добавлена кнопка range
    keyboard = [
        [InlineKeyboardButton("Повторить прогноз",         callback_data="repeat")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")],
        [InlineKeyboardButton("Оставить отзыв",            callback_data="feedback")],
        [InlineKeyboardButton("Выход",                      callback_data="exit")]
    ]
    await q.message.reply_text('Что дальше?', reply_markup=InlineKeyboardMarkup(keyboard))

    context.user_data.pop('range_start', None)
    context.user_data.pop('range_end', None)
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
    # Отправляем на Gmail (заранее заданы переменные окружения GMAIL_USER и GMAIL_PASS)
    user = os.getenv("GMAIL_USER")
    pwd  = os.getenv("GMAIL_PASS")
    msg  = MIMEText(text, _charset="utf-8")
    msg["Subject"] = "Отзыв DemandForecastBot"
    msg["From"]    = user
    msg["To"]      = user
    with SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(user, pwd)
        smtp.send_message(msg)
    await update.message.reply_text("Спасибо за ваш отзыв! Для перезапуска нажмите /start")
    return ConversationHandler.END


async def repeat_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # callback — отвечаем, чтобы убрать “часики”
    q = update.callback_query
    await q.answer()

    # чищаем прошлый контекст
    context.user_data.clear()

    # запускаем start опять, он подхватит update.callback_query.message
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
            SELECT_MODEL:       [CallbackQueryHandler(model_chosen)],
            SELECT_OIL_SOURCE:  [CallbackQueryHandler(oil_source_chosen)],
            ASK_NUM_CRISES:     [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_num_crises)],
            SELECT_CRISIS_TYPE: [CallbackQueryHandler(select_crisis_type)],
            CALENDAR_START:     [CallbackQueryHandler(calendar_start)],
            CALENDAR_END:       [CallbackQueryHandler(calendar_end)],
            INPUT_INTENSITY:    [MessageHandler(filters.TEXT & ~filters.COMMAND, input_intensity)],
            AFTER_FORECAST:     [
                CallbackQueryHandler(repeat_forecast, pattern='^repeat$'),
                CallbackQueryHandler(range_start,    pattern='^range$'),
                CallbackQueryHandler(feedback_start, pattern='^feedback$'),
                CallbackQueryHandler(exit_bot,       pattern='^exit$'),
            ],
            RANGE_START:        [CallbackQueryHandler(range_end)],
            RANGE_END:          [CallbackQueryHandler(range_end)],
            FEEDBACK:           [MessageHandler(filters.TEXT & ~filters.COMMAND, feedback_received)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        allow_reentry=True
    )
    app.add_handler(conv)
    app.run_polling()
