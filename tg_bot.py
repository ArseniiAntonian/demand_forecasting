import logging
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import date

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    ConversationHandler, MessageHandler, filters, ContextTypes
)
from telegram_bot_calendar import DetailedTelegramCalendar

from Ignat_prophet.NEW_predict import forecast_prophet
from belG import huy as tcn_forecast
from arsen.gb_forecast import forecast_lgb

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
    RANGE_END
) = range(10)

MODEL_OPTIONS = {
    'prophet': ('Prophet',    forecast_prophet),
    'tcn':     ('TCN',        tcn_forecast),
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
    kb = [[InlineKeyboardButton(name, callback_data=key)]
          for key, (name, _) in MODEL_OPTIONS.items()]
    await update.message.reply_text(
        'Выберите модель прогнозирования:',
        reply_markup=InlineKeyboardMarkup(kb)
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
    result, cal, _ = DetailedTelegramCalendar(
        min_date=date(2025,1,1),
        max_date=date(2030,12,31)
    ).process(q.data)
    if not result and cal:
        await q.edit_message_reply_markup(reply_markup=cal)
        return CALENDAR_START
    await q.edit_message_text(f'Дата начала: {result}')
    context.user_data['crises'][-1]['start'] = result.isoformat()
    cal, _ = DetailedTelegramCalendar(
        min_date=date(2025,1,1),
        max_date=date(2030,12,31)
    ).build()
    await q.message.reply_text(
        'Выберите дату окончания:',
        reply_markup=cal
    )
    return CALENDAR_END

async def calendar_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    result, cal, _ = DetailedTelegramCalendar(
        min_date=date(2025,1,1),
        max_date=date(2030,12,31)
    ).process(q.data)
    if not result and cal:
        await q.edit_message_reply_markup(reply_markup=cal)
        return CALENDAR_END
    await q.edit_message_text(f'Дата окончания: {result}')
    context.user_data['crises'][-1]['end'] = result.isoformat()
    await q.message.reply_text('Введите интенсивность в процентах (0-100):')
    return INPUT_INTENSITY

async def input_intensity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text.strip().rstrip('%')
        val = float(text)
        assert 0.0 <= val <= 100.0
    except:
        await update.message.reply_text('Введите число от 0 до 100.')
        return INPUT_INTENSITY
    intensity = val / 100.0
    context.user_data['crises'][-1]['intensity'] = intensity
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
    msg = update.callback_query.message if update.callback_query else update.message

    # 1) Load historical data
    df_hist = pd.read_csv('data/ML_with_crisis.csv', parse_dates=['Date'])
    last_oil = df_hist['Oil_Price'].iloc[-1]

    # 2) Build exogenous DataFrame for 2025–2030
    dates = pd.date_range('2025-01-01', '2030-12-01', freq='MS')
    ints = pd.Series(0.0, index=dates)
    for c in context.user_data.get('crises', []):
        start = pd.to_datetime(c['start']); end = pd.to_datetime(c['end'])
        mask = (dates >= start) & (dates <= end)
        ints.loc[mask] = c['intensity']
    df_exog = pd.DataFrame({
        'Date': dates,
        'Oil_Price': last_oil,
        'crisis_intensity': ints.values,
        'crisis_shock': 0.0,
    })

    # 3) Forecast
    key, func = context.user_data['model_key'], None
    _, func = MODEL_OPTIONS[key]
    if key == 'prophet':
        _, result = func(df_exog)
        df_pred = result.rename(columns={'yhat_exp':'Forecast'}).set_index('Date')[['Forecast']]
    else:
        # filter out already-seen dates
        df_train_full = pd.read_csv('data/ML.csv', parse_dates=['Date'])
        last_hist = df_train_full['Date'].max()
        df_new = df_exog[df_exog['Date'] > last_hist].copy()
        df_raw = func(df_new)
        if 'Date' not in df_raw.columns: df_raw['Date'] = df_raw.index
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_pred = df_raw.set_index('Date')[['Forecast']]

    # 4) Plot history (2003–2025) + forecast (2025–2030) on 0–2000 scale, larger rectangle
    fig, ax = plt.subplots(figsize=(14, 6))  # wider rectangular
    ax.plot(df_hist['Date'], df_hist['Freight_Price'], label='История 2003–2025')
    ax.plot(df_pred.index, df_pred['Forecast'], '--', label='Прогноз 2025–2030')
    ax.set_ylim(0, 2200)
    ax.set_title('Тест 2003–2025 и прогноз 2025–2030')
    ax.legend()
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    await msg.reply_photo(photo=buf)

    # 5) Buttons
    keyboard = [
        [InlineKeyboardButton("Повторить прогноз", callback_data="repeat")],
        [InlineKeyboardButton("Выход",             callback_data="exit")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")]
    ]
    await msg.reply_text(
        'Что дальше?',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return AFTER_FORECAST



async def range_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    cal, _ = DetailedTelegramCalendar(min_date=date(2025,1,1), max_date=date(2030,12,31)).build()
    await q.message.reply_text('Выберите начало диапазона прогноза:', reply_markup=cal)
    return RANGE_START

async def range_end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    res, cal, _ = DetailedTelegramCalendar(min_date=date(2025,1,1), max_date=date(2030,12,31)).process(q.data)
    if not res and cal:
        await q.edit_message_reply_markup(reply_markup=cal)
        return RANGE_END
    if 'range_start' not in context.user_data:
        context.user_data['range_start'] = res
        await q.edit_message_text(f'Начало диапазона: {res}')
        cal, _ = DetailedTelegramCalendar(min_date=date(2025,1,1), max_date=date(2030,12,31)).build()
        await q.message.reply_text('Выберите конец диапазона прогноза:', reply_markup=cal)
        return RANGE_END
    context.user_data['range_end'] = res
    await q.edit_message_text(f'Конец диапазона: {res}')

    df_pred = context.user_data['df_pred']
    start = pd.to_datetime(context.user_data['range_start'])
    end = pd.to_datetime(context.user_data['range_end'])
    df_slice = df_pred[(df_pred.index >= start) & (df_pred.index <= end)]
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_slice.index, df_slice['Forecast'], '-', label='Прогноз')
    ax.set_ylim(0, 1000)
    ax.set_title(f'Прогноз с {start.date()} по {end.date()}')
    ax.grid(True)
    fig.savefig(buf, format='png'); buf.seek(0)
    await q.message.reply_photo(photo=buf)

    keyboard = [
        [InlineKeyboardButton("Повторить прогноз", callback_data="repeat")],
        [InlineKeyboardButton("Выход", callback_data="exit")],
        [InlineKeyboardButton("Выбрать диапазон прогноза", callback_data="range")]
    ]
    await q.message.reply_text('Что дальше?', reply_markup=InlineKeyboardMarkup(keyboard))
    context.user_data.pop('range_start', None)
    context.user_data.pop('range_end', None)
    return AFTER_FORECAST

async def repeat_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    context.user_data.clear()
    kb = [[InlineKeyboardButton(name, callback_data=key)]
          for key,(name,_) in MODEL_OPTIONS.items()]
    await q.message.reply_text(
        'Выберите модель прогнозирования:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_MODEL

async def exit_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    await q.message.reply_text(
        'Бот завершает работу. До встречи! Для перезапуска нажмите команду /start'
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
                CallbackQueryHandler(exit_bot,       pattern='^exit$'),
                CallbackQueryHandler(range_start,    pattern='^range$')
            ],
            RANGE_START:        [CallbackQueryHandler(range_end)],
            RANGE_END:          [CallbackQueryHandler(range_end)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    app.add_handler(conv)
    app.run_polling()
