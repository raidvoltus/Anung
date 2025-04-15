import os
import time
import pickle
import joblib
import requests
import logging
import numpy as np
import pandas as pd
import random
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf
import matplotlib.pyplot as plt

from ta import momentum, trend, volatility, volume
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler()
    ]
)

# Konfigurasi
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

model_cls_path = "model_cls.txt"
model_high_path = "model_high.txt"
model_low_path = "model_low.txt"
model_lstm_path = "model_lstm.h5"

BACKUP_CSV_PATH = "stock_data_backup.csv"
SCALER_PATH = "scaler_target.pkl"

TP_MULTIPLIER = 0.01
SL_MULTIPLIER = 0.015
MIN_PROBABILITY = 0.045
RETRAIN_INTERVAL = 3

yf.set_tz_cache_location("/tmp/py-yfinance-cache")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

STOCK_LIST = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "UNVR.JK"
]

# Fungsi Telegram
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.error("Telegram token/chat_id tidak ditemukan.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# Cache untuk menghindari tekanan pada yfinance dan meningkatkan kecepatan
session = requests_cache.CachedSession('yfinance_cache', expire_after=300)

def get_stock_data(ticker, retries=5, delay=2):
    for attempt in range(1, retries + 1):
        try:
            stock = yf.Ticker(ticker, session=session)
            data = stock.history(period="730d", interval="1h", timeout=10)
            
            if data.empty or len(data) < 200:
                logging.warning(f"{ticker} - Data kosong atau terlalu sedikit (len={len(data)}), attempt {attempt}")
                time.sleep(delay + random.uniform(0, 2))
                continue

            if "Volume" not in data.columns:
                data["Volume"] = 0

            data["ticker"] = ticker
            return data

        except Exception as e:
            logging.error(f"{ticker} - Attempt {attempt}/{retries} gagal: {e}")
            time.sleep(delay + random.uniform(0, 2))

    logging.error(f"{ticker} - Gagal mengambil data setelah {retries} attempt.")
    return None

# Indikator
def calculate_indicators(df):
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    macd = trend.MACD(df["Close"], window_slow=13, window_fast=5, window_sign=5)
    df["MACD"] = macd.macd()
    df["MACD_Hist"] = macd.macd_diff()
    bb = volatility.BollingerBands(df["Close"])
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["Support"] = df["Low"].rolling(24).min()
    df["Resistance"] = df["High"].rolling(24).max()
    df["RSI"] = momentum.RSIIndicator(df["Close"]).rsi()
    df["SMA_20"] = trend.SMAIndicator(df["Close"], 20).sma_indicator()
    df["SMA_50"] = trend.SMAIndicator(df["Close"], 50).sma_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["future_high"] = df["High"].shift(-6)
    df["future_low"] = df["Low"].shift(-6)
    return df.dropna()

# Training model
def train_lightgbm(X, y):
    model = lgb.LGBMRegressor()
    model.fit(X, y)
    return model

def train_classifier(X, y):
    model = lgb.LGBMClassifier()
    model.fit(X, y)
    return model

def train_lstm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(units=64, input_shape=(1, X_scaled.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_lstm, y, epochs=10, batch_size=32, verbose=0)
    return model

# Analisis per saham
def analyze_stock(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None

    df = calculate_indicators(df)

    features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_20", "SMA_50",
                "VWAP", "ADX", "BB_Upper", "BB_Lower", "Support", "Resistance"]
    df = df.dropna(subset=features + ["future_high", "future_low"])
    if df.empty:
        return None

    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]
    y_binary = (df["future_high"] > df["Close"]).astype(int)

    X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2)
    y_train_low = y_low[X_train.index]
    y_train_cls = y_binary[X_train.index]

    # Path model berdasarkan ticker
    model_high_path = f"models/{ticker}_model_high.txt"
    model_low_path = f"models/{ticker}_model_low.txt"
    model_cls_path = f"models/{ticker}_model_cls.pkl"
    model_lstm_path = f"models/{ticker}_model_lstm.h5"

    retrain = not all([os.path.exists(p) for p in [model_high_path, model_low_path, model_cls_path, model_lstm_path]])

    try:
        if retrain:
            model_high = train_lightgbm(X_train, y_train_high)
            model_low = train_lightgbm(X_train, y_train_low)
            model_cls = train_classifier(X_train, y_train_cls)
            model_lstm = train_lstm(X_train, y_train_high)

            joblib.dump(model_high, model_high_path)
            joblib.dump(model_low, model_low_path)
            joblib.dump(model_cls, model_cls_path)
            model_lstm.save(model_lstm_path)
        else:
            model_high = joblib.load(model_high_path)
            model_low = joblib.load(model_low_path)
            model_cls = joblib.load(model_cls_path)
            model_lstm = load_model(model_lstm_path)

            # Validasi jumlah fitur
            if hasattr(model_high, 'n_features_in_') and model_high.n_features_in_ != len(X.columns):
                logging.warning(f"[{ticker}] Jumlah fitur tidak cocok. Model dilatih dengan {model_high.n_features_in_}, input sekarang {len(X.columns)}. Retraining...")
                os.remove(model_high_path)
                os.remove(model_low_path)
                os.remove(model_cls_path)
                os.remove(model_lstm_path)
                return analyze_stock(ticker)  # Retry dengan retrain
    except Exception as e:
        logging.error(f"Error saat memuat atau melatih model untuk {ticker}: {e}")
        return None

    predicted_high = model_high.predict(X)
    predicted_low = model_low.predict(X)
    predicted_class = model_cls.predict(X)

    scaler = joblib.load(SCALER_PATH)
    X_lstm_scaled = scaler.transform(X)
    X_lstm_scaled = np.reshape(X_lstm_scaled, (X_lstm_scaled.shape[0], 1, X_lstm_scaled.shape[1]))
    predicted_lstm = model_lstm.predict(X_lstm_scaled, verbose=0)

    df["predicted_high"] = predicted_high
    df["predicted_low"] = predicted_low
    df["predicted_class"] = predicted_class
    df["predicted_lstm"] = predicted_lstm

    latest = df.iloc[-1]
    harga = latest["Close"]
    tp = float(latest["predicted_high"])
    sl = float(latest["predicted_low"])
    prob = float(latest["predicted_class"])
    aksi = "buy" if prob == 1 else "sell"
    potensi = round(((tp - harga) / harga) * 100, 2)

    if aksi == "buy" and potensi >= 1.5 and prob >= 0.7:
        return {
            "ticker": ticker,
            "harga": harga,
            "stop_loss": sl,
            "take_profit": tp,
            "aksi": aksi,
            "probability": prob,
            "profit_pct": potensi
        }
    return None

# Plot distribusi
def plot_probability_distribution(results):
    try:
        df = pd.DataFrame(results)
        plt.figure(figsize=(10, 4))
        plt.hist(df["probability"], bins=10, color='skyblue')
        plt.title("Distribusi Probabilitas")
        plt.savefig("probability_distribution.png")
    except Exception as e:
        logging.warning(f"Gagal membuat plot: {e}")

# Eksekusi utama
if __name__ == "__main__":
    logging.info("Mulai analisis saham...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r]

    if results:
        plot_probability_distribution(results)
        top_5 = sorted(results, key=lambda x: x["take_profit"], reverse=True)[:5]
        msg = "<b>Sinyal Saham Terbaik Hari Ini:</b>\n"
        for r in top_5:
            msg += (
                f"\n🔹 {r['ticker']}\n"
                f"   Harga: {r['harga']:.2f}\n"
                f"   TP: {r['take_profit']:.2f}\n"
                f"   SL: {r['stop_loss']:.2f}\n"
                f"   Aksi: <b>{r['aksi'].upper()}</b>\n"
                f"   Potensi: {r['profit_pct']}%\n"
                f"   Probabilitas: {r['probability']*100:.2f}%\n"
            )
        send_telegram_message(msg)
    else:
        logging.info("Tidak ada sinyal valid.")

    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
