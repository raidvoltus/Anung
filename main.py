import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import telegram
import logging
import time
import os
from datetime import datetime, timedelta

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Parameter
TP_MULTIPLIER = 1.5
SL_MULTIPLIER = 1.0
MIN_PROBABILITY = 0.7
RETRAIN_INTERVAL_DAYS = 7

# Telegram
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

bot = telegram.Bot(token=TELEGRAM_TOKEN)

# Ambil daftar saham dari IDX
def get_stock_list():
    url = "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        tickers = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if cols:
                ticker = cols[0].text.strip()
                tickers.append(f"{ticker}.JK")
        return tickers
    except Exception as e:
        logging.warning(f"Gagal mengambil daftar saham dari situs IDX: {e}")
        return []

# Ambil data harga
def fetch_data(symbol):
    try:
        df = yf.download(symbol, period='6mo', interval='1d', progress=False)
        if df.empty or len(df) < 30:
            raise ValueError("Data terlalu sedikit atau kosong")
        df['Return'] = df['Close'].pct_change()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.warning(f"Gagal mengambil data untuk {symbol}: {e}")
        return None

def save_historical_data(df, ticker):
    filename = f"data_historis/{ticker}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

# Fitur dan label
def generate_features(df):
    df['Target_High'] = (df['High'].shift(-3) > df['Close'] * (1 + TP_MULTIPLIER / 100)).astype(int)
    df['Target_Low'] = (df['Low'].shift(-3) < df['Close'] * (1 - SL_MULTIPLIER / 100)).astype(int)
    features = df[['Return', 'MA20', 'MA50']]
    return features.dropna(), df['Target_High'].dropna(), df['Target_Low'].dropna()

# Training model
def train_lightgbm(X_train, y_train):
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model

# Model LSTM
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Prediksi saham
def analyze_stock(symbol):
    df = fetch_data(symbol)
    if df is None:
        return None

    X, y_high, y_low = generate_features(df)
    if len(X) < 30:
        return None

    X_train, X_test, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, shuffle=False)
    X_train, X_test, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, shuffle=False)

    model_high = train_lightgbm(X_train, y_train_high)
    model_low = train_lightgbm(X_train, y_train_low)

    last_features = X.iloc[-1:]
    prob_high = model_high.predict_proba(last_features)[0][1]
    prob_low = model_low.predict_proba(last_features)[0][1]

    probas = model_high.predict_proba(X_last)
    max_proba = max(probas[0])
    if max_proba < 0.7:
        stop_loss = close_price * (1 - SL_MULTIPLIER / 100)
        return {
            'symbol': symbol,
            'action': 'BUY',
            'close': round(close_price, 2),
            'tp': round(target_price, 2),
            'sl': round(stop_loss, 2),
            'prob': round(prob_high, 2),
            'potential_profit': round((target_price - close_price) / close_price * 100, 2)
        }
    return None

df = calculate_indicators(df)
save_historical_data(df, ticker)

# Self-learning model evaluation (dummy version)
def evaluate_and_retrain():
    now = datetime.now()
    if not os.path.exists('last_retrain.txt'):
        with open('last_retrain.txt', 'w') as f:
            f.write(now.strftime('%Y-%m-%d'))
    else:
        with open('last_retrain.txt', 'r') as f:
            last = datetime.strptime(f.read().strip(), '%Y-%m-%d')
        if (now - last).days >= RETRAIN_INTERVAL_DAYS:
            logging.info("Retraining model secara otomatis...")
            with open('last_retrain.txt', 'w') as f:
                f.write(now.strftime('%Y-%m-%d'))

# Kirim sinyal via Telegram
def send_telegram_message(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        logging.warning(f"Gagal mengirim pesan Telegram: {e}")

# MAIN
def main():
    logging.info("Memulai bot trading saham...")
    stock_list = get_stock_list()
    if not stock_list:
        logging.warning("Gagal mengambil daftar saham.")
        return

    signals = []
    for symbol in stock_list:
        signal = analyze_stock(symbol)
        if signal:
            signals.append(signal)

    signals = sorted(signals, key=lambda x: x['potential_profit'], reverse=True)[:5]
    if signals:
        messages = []
        for s in signals:
            messages.append(
                f"{s['symbol']} | {s['action']} @ {s['close']}\nTP: {s['tp']} | SL: {s['sl']}\nProfit: {s['potential_profit']}% | Prob: {s['prob']}"
            )
        send_telegram_message("\n\n".join(messages))
    else:
        send_telegram_message("Tidak ada sinyal beli hari ini.")

    evaluate_and_retrain()

if __name__ == "__main__":
    main()
