import os
import time
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

# === Konfigurasi ===
MODEL_HIGH_PATH = "model_high.pkl"
MODEL_LOW_PATH = "model_low.pkl"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Telegram Bot
TELEGRAM_TOKEN = "ISI_TOKEN_ANDA"
TELEGRAM_CHAT_ID = "ISI_CHAT_ID_ANDA"

# === Ambil daftar saham dari situs IDX ===
def get_stock_list():
    url = "https://www.idx.co.id/en-us/listed-companies/company-profiles/"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        tickers = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                ticker = cols[1].text.strip()
                if ticker and ticker.isalpha():
                    tickers.append(ticker + ".JK")
        return tickers
    except Exception as e:
        logging.warning(f"Gagal mengambil daftar saham dari situs IDX: {e}")
        return []

# === Ambil data saham dari yfinance ===
def download_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df if not df.empty else None
    except Exception as e:
        logging.warning(f"Error download {ticker}: {e}")
        return None

# === Ekstrak fitur teknikal ===
def extract_features(df):
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["STD5"] = df["Close"].rolling(window=5).std()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["Volume_Change"] = df["Volume"].pct_change()
    df = df.dropna()
    return df

# === Hitung RSI ===
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

# === Latih model LightGBM ===
def train_lightgbm(X, y):
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model

# === Kirim pesan ke Telegram ===
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        logging.error(f"Gagal kirim pesan Telegram: {e}")

# === Analisa Saham ===
def analyze_stock(ticker):
    df = download_stock_data(ticker)
    if df is None or len(df) < 30:
        return None

    df = extract_features(df)
    if len(df) < 30:
        return None

    df["Target_High"] = df["High"].shift(-1)
    df["Target_Low"] = df["Low"].shift(-1)
    df.dropna(inplace=True)

    feature_cols = ["Close", "MA5", "MA10", "STD5", "RSI", "Volume_Change"]
    X = df[feature_cols]
    y_high = df["Target_High"]
    y_low = df["Target_Low"]

    X_train, _, y_train_high, y_train_low = train_test_split(X, y_high, test_size=0.2, shuffle=False), \
                                            df["Target_High"], df["Target_Low"]

    model_high = train_lightgbm(X_train, y_train_high)
    model_low = train_lightgbm(X_train, y_train_low)

    joblib.dump(model_high, MODEL_HIGH_PATH)
    joblib.dump(model_low, MODEL_LOW_PATH)

    last_data = X.iloc[-1:]
    predicted_high = model_high.predict(last_data)[0]
    predicted_low = model_low.predict(last_data)[0]
    current_price = df["Close"].iloc[-1]

    predicted_profit_pct = ((predicted_high - current_price) / current_price) * 100
    predicted_risk_pct = ((current_price - predicted_low) / current_price) * 100

    predicted_range = predicted_high - predicted_low
    if predicted_range <= 0:
        return None

    probability = min(max(predicted_profit_pct / predicted_range, 0), 1)

    return {
        "ticker": ticker,
        "action": "BUY",
        "price": round(current_price, 2),
        "TP": round(predicted_high, 2),
        "SL": round(predicted_low, 2),
        "potential_profit": round(predicted_profit_pct, 2),
        "probability": round(probability, 2)
    }

# === Kirim Sinyal Top 5 ===
def send_top_signals(signals):
    signals = [s for s in signals if s and s["probability"] > 0.7]
    top_signals = sorted(signals, key=lambda x: x["potential_profit"], reverse=True)[:5]

    if not top_signals:
        send_telegram_message("Tidak ada sinyal dengan probabilitas > 0.7 hari ini.")
        return

    message = "<b>Top 5 Sinyal Saham Hari Ini (Prob > 0.7)</b>\n\n"
    for s in top_signals:
        message += (
            f"<b>{s['ticker']}</b> - {s['action']}\n"
            f"Harga: {s['price']} | TP: {s['TP']} | SL: {s['SL']}\n"
            f"Potensi Profit: {s['potential_profit']}% | Prob: {s['probability']}\n\n"
        )
    send_telegram_message(message)

# === Fungsi Utama ===
def main():
    logging.info("Memulai bot trading saham...")
    stock_list = get_stock_list()
    if not stock_list:
        logging.warning("Gagal mengambil daftar saham.")
        return

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(analyze_stock, stock_list))

    send_top_signals(results)

if __name__ == "__main__":
    main()
