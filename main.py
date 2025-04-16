# === Import Library ===
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
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
ATR_MULTIPLIER = 2.5
RETRAIN_INTERVAL = 7
MODEL_HIGH_PATH = "model_high.pkl"
MODEL_LOW_PATH = "model_low.pkl"
MODEL_PROB_PATH = "model_prob.pkl"
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

# === Kirim Pesan Telegram ===
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

# === Ambil Data Saham ===
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data is not None and not data.empty and len(data) > 100:
            data["ticker"] = ticker
            return data
        logger.warning(f"Data kosong/kurang: {ticker}")
    except Exception as e:
        logger.error(f"Error data {ticker}: {e}")
    return None

# === Hitung Indikator Teknikal ===
def calculate_indicators(df):
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
    df["profit_pct"] = (df["future_high"] - df["Close"]) / df["Close"] * 100
    df["label"] = (df["profit_pct"] >= MIN_PROFIT_PERCENT).astype(int)
    return df.dropna()

# === Pelatihan Model ===
def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03)
    model.fit(X, y)
    return model

def train_classifier(X, y):
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
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
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime > timedelta(days=interval_days)

# === Analisis Satu Saham ===
def analyze_stock(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None
    df = calculate_indicators(df)
    features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_50", "SMA_200",
                "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
    df = df.dropna(subset=features + ["future_high", "future_low", "label"])
    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]
    y_label = df["label"]

    retrain = is_retraining_needed(MODEL_HIGH_PATH)

    if retrain:
        model_high = train_lightgbm(X, y_high)
        model_low = train_lightgbm(X, y_low)
        model_prob = train_classifier(X, y_label)
        model_lstm = train_lstm(X, y_high)
        joblib.dump(model_high, MODEL_HIGH_PATH)
        joblib.dump(model_low, MODEL_LOW_PATH)
        joblib.dump(model_prob, MODEL_PROB_PATH)
        model_lstm.save(MODEL_LSTM_PATH)
    else:
        model_high = joblib.load(MODEL_HIGH_PATH)
        model_low = joblib.load(MODEL_LOW_PATH)
        model_prob = joblib.load(MODEL_PROB_PATH)

    X_last = X.iloc[[-1]]
    current_price = df["Close"].iloc[-1]
    pred_high = model_high.predict(X_last)[0]
    pred_low = model_low.predict(X_last)[0]
    prob = model_prob.predict_proba(X_last)[0][1]  # Probabilitas "profit >= x%"

    profit_percent = (pred_high - current_price) / current_price * 100

    if prob < MIN_PROBABILITY or profit_percent < MIN_PROFIT_PERCENT:
        return None

    return {
        "ticker": ticker,
        "harga": round(current_price, 2),
        "take_profit": round(pred_high, 2),
        "stop_loss": round(pred_low, 2),
        "aksi": "beli",
        "prob": round(prob, 2),
        "profit_pct": round(profit_percent, 2)
    }

# === Main Eksekusi ===
if __name__ == "__main__":
    logger.info("Memulai analisis saham...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r and r["aksi"] == "beli"]
    top5 = sorted(results, key=lambda x: x["profit_pct"], reverse=True)[:5]

    if top5:
        message = "<b>Top 5 Sinyal Trading Hari Ini</b>\n"
        for s in top5:
            message += (f"\n<b>{s['ticker']}</b>\nHarga: {s['harga']}\nTP: {s['take_profit']}\n"
                        f"SL: {s['stop_loss']}\nPotensi: {s['profit_pct']}%\nProb: {s['prob']*100:.1f}%\n"
                        f"Aksi: <b>{s['aksi'].upper()}</b>\n")
        send_telegram_message(message)

    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logger.info("Analisis selesai. Data disimpan.")
