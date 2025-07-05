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
    AFTER_FORECAST
) = range(8)

MODEL_OPTIONS = {
    'prophet': ('Prophet',    forecast_prophet),
    'tcn':     ('TCN',        tcn_forecast),
    'gb':      ('LightGBM',   forecast_lgb),
}

CRISIS_TYPES = ['Financial', 'Pandemic', 'Geopolitical', 'Natural', 'Logistical']


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
    await q.message.reply_text('Сколько кризисов задать? (0–5)')
    return ASK_NUM_CRISES


async def ask_num_crises(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        n = int(update.message.text.strip())
        assert 0 <= n <= 5
    except:
        await update.message.reply_text('Введите число от 0 до 5.')
        return ASK_NUM_CRISES
    context.user_data['num_crises'] = n
    context.user_data['crises'] = []
    if n == 0:
        return await launch_forecast(update, context)
    context.user_data['current'] = 1
    kb = [[InlineKeyboardButton(t, callback_data=t)] for t in CRISIS_TYPES]
    await update.message.reply_text(
        'Выберите тип кризиса #1:',
        reply_markup=InlineKeyboardMarkup(kb)
    )
    return SELECT_CRISIS_TYPE


async def select_crisis_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    ctype = q.data
    idx = context.user_data['current']
    context.user_data['crises'].append({'type': ctype})
    cal, _ = DetailedTelegramCalendar(
        min_date=date(2025,1,1),
        max_date=date(2030,12,31)
    ).build()
    await q.message.reply_text(
        f'Выберите дату начала для кризиса #{idx} ({ctype}):',
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
    await q.message.reply_text('Введите интенсивность (0.0–1.0):')
    return INPUT_INTENSITY


async def input_intensity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        val = float(update.message.text.strip()); assert 0.0 <= val <= 1.0
    except:
        await update.message.reply_text('Введите число от 0.0 до 1.0')
        return INPUT_INTENSITY
    context.user_data['crises'][-1]['intensity'] = val
    cur = context.user_data['current']
    total = context.user_data['num_crises']
    if cur < total:
        context.user_data['current'] += 1
        kb = [[InlineKeyboardButton(t, callback_data=t)] for t in CRISIS_TYPES]
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

    # 1) Загружаем исторические данные для бот-кейса
    df_hist = pd.read_csv('data/ML_with_crisis.csv', parse_dates=['Date'])
    last_oil = df_hist['Oil_Price'].iloc[-1]

    # 2) Генерируем даты прогноза и интенсивности
    dates = pd.date_range('2025-01-01', '2030-12-01', freq='MS')
    ints = pd.Series(0.0, index=dates)
    for c in context.user_data.get('crises', []):
        start = pd.to_datetime(c['start'])
        end   = pd.to_datetime(c['end'])
        mask = (dates >= start) & (dates <= end)
        ints.loc[mask] = c['intensity']

    # 3) Формируем exog dataframe
    df_exog = pd.DataFrame({
        'Date': dates,
        'Oil_Price': last_oil,
        'crisis_intensity': ints.values,
        'crisis_shock': 0.0,
    })

    # 4) Выбираем модель
    key, func = context.user_data['model_key'], None
    _, func = MODEL_OPTIONS[key]

    if key == 'prophet':
        train, result = func(df_exog)
        df_pred = (
            result
            .rename(columns={'yhat_exp': 'Forecast'})
            .set_index('Date')[['Forecast']]
        )
    else:
        # ----------------------------------------
        # ОТСЕКАЕМ ДАТЫ, УЖЕ ЕСТЬ В ИСТОРИИ ML.csv
        df_train_full = pd.read_csv('data/ML.csv', parse_dates=['Date'])
        last_hist_date = df_train_full['Date'].max()

        df_exog_filtered = df_exog[df_exog['Date'] > last_hist_date].copy()
        # Передаём в forecast_lgb только новые даты
        df_pred_raw = func(df_exog_filtered)

        # Вытаскиваем Forecast и Date
        if 'Date' not in df_pred_raw.columns:
            df_pred_raw['Date'] = df_pred_raw.index
        df_pred_raw['Date'] = pd.to_datetime(df_pred_raw['Date'])
        df_pred = df_pred_raw.set_index('Date')[['Forecast']]
        # ----------------------------------------

    # 5) Рисуем только прогноз
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1000)
    ax.plot(df_pred.index, df_pred['Forecast'], '-', label='Прогноз')
    ax.set_title('Прогноз 2025–2030')

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    await msg.reply_photo(photo=buf)

    # 6) Кнопки «Повторить» / «Выход»
    keyboard = [
        [InlineKeyboardButton("Повторить прогноз", callback_data="repeat")],
        [InlineKeyboardButton("Выход",             callback_data="exit")]
    ]
    await msg.reply_text(
        'Что дальше?',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
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
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    app.add_handler(conv)
    app.run_polling()
