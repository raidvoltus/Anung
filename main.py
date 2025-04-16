# === trading_bot.py ===
import os, time, joblib, requests, logging
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf
from datetime import datetime, timedelta
from ta import momentum, trend, volatility, volume
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

# === Konfigurasi ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RETRAIN_INTERVAL = 7
MODEL_HIGH_PATH = "model_high.pkl"
MODEL_LOW_PATH = "model_low.pkl"
MODEL_LSTM_PATH = "model_lstm.keras"
BACKUP_CSV_PATH = "stock_data_backup.csv"
MAX_WORKERS = 5
MIN_PROBABILITY = 0.7
MIN_PROFIT_PERCENT = 2.0

# === Logging ===
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = RotatingFileHandler("trading.log", maxBytes=2_000_000, backupCount=3)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# === Daftar Saham ===
STOCK_LIST = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK", "ANTM.JK"]

def send_telegram_message(message, max_retries=3):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    for _ in range(max_retries):
        try:
            response = requests.post(url, data=data, timeout=10)
            if response.ok:
                return True
        except Exception as e:
            logger.warning(f"Gagal kirim Telegram: {e}")
        time.sleep(2)
    return False

def get_stock_data(ticker, max_retries=3):
    for _ in range(max_retries):
        try:
            data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if data is not None and not data.empty and len(data) > 100:
                data["ticker"] = ticker
                return data
        except Exception as e:
            logger.error(f"Gagal ambil data {ticker}: {e}")
        time.sleep(2)
    return None

def calculate_indicators(df):
    try:
        df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=10).average_true_range()
        df["RSI"] = momentum.RSIIndicator(df["Close"], window=10).rsi()
        macd = trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Hist"] = macd.macd_diff()
        df["SMA_50"] = trend.SMAIndicator(df["Close"], window=20).sma_indicator()
        df["SMA_200"] = trend.SMAIndicator(df["Close"], window=50).sma_indicator()
        bb = volatility.BollingerBands(df["Close"])
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
        df["Support"] = df["Low"].rolling(20).min()
        df["Resistance"] = df["High"].rolling(20).max()
        df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
        df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
        df["future_high"] = df["High"].shift(-1)
        df["future_low"] = df["Low"].shift(-1)
        return df.dropna()
    except Exception as e:
        logger.error(f"Error hitung indikator: {e}")
        return None

def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03)
    model.fit(X, y)
    return model

def train_lstm(X, y):
    X = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=25, batch_size=32, verbose=0)
    return model

def is_retraining_needed(path, interval_days=RETRAIN_INTERVAL):
    if not os.path.exists(path):
        return True
    last_trained = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - last_trained > timedelta(days=interval_days)

def analyze_stock(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None
    df = calculate_indicators(df)
    if df is None or df.empty:
        logger.warning(f"Data indikator kosong: {ticker}")
        return None

    features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
    df = df.dropna(subset=features + ["future_high", "future_low"])
    if df.empty:
        return None

    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]

    retrain = is_retraining_needed(MODEL_HIGH_PATH)
    if retrain:
        logger.info(f"Retrain model untuk {ticker}")
        model_high = train_lightgbm(X, y_high)
        model_low = train_lightgbm(X, y_low)
        model_lstm = train_lstm(X, y_high)
        joblib.dump(model_high, MODEL_HIGH_PATH)
        joblib.dump(model_low, MODEL_LOW_PATH)
        model_lstm.save(MODEL_LSTM_PATH)
    else:
        model_high = joblib.load(MODEL_HIGH_PATH)
        model_low = joblib.load(MODEL_LOW_PATH)

    X_last = X.iloc[[-1]]
    current_price = df["Close"].iloc[-1]
    pred_high = model_high.predict(X_last)[0]
    pred_low = model_low.predict(X_last)[0]

    profit_pct = (pred_high - current_price) / current_price * 100
    probability = np.clip((pred_high - pred_low) / (df["High"].max() - df["Low"].min()), 0, 1)

    if profit_pct < MIN_PROFIT_PERCENT or probability < MIN_PROBABILITY:
        return None

    return {
        "ticker": ticker,
        "harga": round(current_price, 2),
        "take_profit": round(pred_high, 2),
        "stop_loss": round(pred_low, 2),
        "potensi_profit": round(profit_pct, 2),
        "probabilitas": round(probability, 2),
        "aksi": "beli"
    }

def main():
    logger.info("Memulai analisis saham...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r]
    top5 = sorted(results, key=lambda x: (x["potensi_profit"], x["probabilitas"]), reverse=True)[:5]

    if top5:
        message = "<b>Top 5 Sinyal Trading Hari Ini</b>\n"
        for s in top5:
            message += (f"\n<b>{s['ticker']}</b>\nHarga: {s['harga']}\nTP: {s['take_profit']}\n"
                        f"SL: {s['stop_loss']}\nProfit: {s['potensi_profit']}%\nProb: {s['probabilitas']}\n"
                        f"Aksi: <b>{s['aksi'].upper()}</b>\n")
        send_telegram_message(message)
    else:
        send_telegram_message("Tidak ada sinyal trading yang memenuhi kriteria hari ini.")

    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logger.info("Analisis selesai. Data disimpan.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Script gagal: {e}")
