import os
import time
import joblib
import requests
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf
import matplotlib.pyplot as plt
from ta import momentum, trend, volatility, volume
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
import pickle

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[RotatingFileHandler("trading.log", maxBytes=5*1024*1024, backupCount=2), logging.StreamHandler()]
)

# --- KONFIGURASI UTAMA ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
ATR_MULTIPLIER = 2.5
RETRAIN_INTERVAL = 7

model_paths = {
    "cls": "model_cls.txt",
    "high": "model_high.txt",
    "low": "model_low.txt",
    "lstm": "model_lstm.h5"
}

log_paths = {
    "backup_csv": "stock_data_backup.csv",
    "predictions_csv": "weekly_predictions.csv",
    "evaluation_txt": "weekly_evaluation.txt"
}

# Daftar saham yang dianalisis
STOCK_LIST = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "ASII.JK", "TLKM.JK"]

# --- UTILITY ---
def send_telegram_message(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("Telegram tidak dikonfigurasi.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        res = requests.post(url, data=data)
        if res.status_code != 200:
            logging.error(f"Gagal kirim Telegram: {res.text}")
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="90d", interval="1h")
        if df is None or df.empty or len(df) < 33:
            return None
        df["Volume"] = df.get("Volume", 0)
        df["ticker"] = ticker
        return df
    except Exception as e:
        logging.error(f"Data Error {ticker}: {e}")
        return None

def calculate_indicators(df):
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    macd = trend.MACD(df["Close"], 13, 5, 5)
    df["MACD"], df["Signal_Line"], df["MACD_Hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
    bb = volatility.BollingerBands(df["Close"], window=20)
    df["BB_Upper"], df["BB_Lower"] = bb.bollinger_hband(), bb.bollinger_lband()
    df["Support"] = df["Low"].rolling(window=48).min()
    df["Resistance"] = df["High"].rolling(window=48).max()
    stoch = momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14)
    df["%K"], df["%D"] = stoch.stoch(), stoch.stoch_signal()
    df["RSI"] = momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["SMA_20"] = trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["SMA_50"] = trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    df["SMA_100"] = trend.SMAIndicator(df["Close"], window=100).sma_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=10).adx()
    df["future_high"] = df["High"].shift(-6)
    df["future_low"] = df["Low"].shift(-6)
    return df.dropna()

def train_lightgbm(X, y, classifier=False):
    model = lgb.LGBMClassifier() if classifier else lgb.LGBMRegressor()
    model.fit(X, y)
    return model

def train_lstm(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    model = Sequential([
        LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=0)
    joblib.dump(scaler, "scaler_lstm.save")
    return model

def analyze_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df is None:
            return
        df = calculate_indicators(df)
        features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_20", "SMA_50", "SMA_100", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
        df.dropna(subset=features + ["future_high", "future_low"], inplace=True)
        X, y_high, y_low = df[features], df["future_high"], df["future_low"]
        y_binary = (df["future_high"] > df["Close"]).astype(int)
        X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2)
        y_train_low = y_low[X_train.index]
        y_train_cls = y_binary[X_train.index]

        retrain = True
        if os.path.exists(model_paths["high"]):
            last_mod = datetime.fromtimestamp(os.path.getmtime(model_paths["high"]))
            retrain = (datetime.now() - last_mod).days >= RETRAIN_INTERVAL

        if retrain:
            joblib.dump(train_lightgbm(X_train, y_train_high), model_paths["high"])
            joblib.dump(train_lightgbm(X_train, y_train_low), model_paths["low"])
            joblib.dump(train_lightgbm(X_train, y_train_cls, classifier=True), model_paths["cls"])
            train_lstm(X_train, y_train_high).save(model_paths["lstm"])

        model_high = joblib.load(model_paths["high"])
        model_low = joblib.load(model_paths["low"])
        model_cls = joblib.load(model_paths["cls"])
        model_lstm = load_model(model_paths["lstm"])
        scaler_lstm = joblib.load("scaler_lstm.save")

        X_scaled = scaler_lstm.transform(X)
        X_seq = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
        pred_high = model_high.predict(X)
        pred_low = model_low.predict(X)
        pred_cls = model_cls.predict(X)
        pred_lstm = model_lstm.predict(X_seq)

        df["predicted_high"] = pred_high
        df["predicted_low"] = pred_low
        df["predicted_class"] = pred_cls
        df["predicted_lstm"] = pred_lstm

        evaluation = {
            "ticker": ticker,
            "mae_high": mean_absolute_error(y_high, pred_high),
            "mae_low": mean_absolute_error(y_low, pred_low),
            "mape_high": mean_absolute_percentage_error(y_high, pred_high),
            "mape_low": mean_absolute_percentage_error(y_low, pred_low)
        }
        with open(EVALUATION_LOG_PATH, "a") as eval_file:
            eval_file.write(str(evaluation) + "\n")

        msg = f"<b>Prediksi {ticker}</b>
        High: {pred_high[-1]:.2f}
        Low: {pred_low[-1]:.2f}
        Class: {'Naik' if pred_cls[-1] == 1 else 'Turun'}
        LSTM: {pred_lstm[-1][0]:.2f}"
        send_telegram_message(msg)

        df.to_csv(log_paths["predictions_csv"], mode='a', header=not os.path.exists(log_paths["predictions_csv"]))

    except Exception as e:
        logging.error(f"Error analisis {ticker}: {e}")

def analyze_multiple_stocks():
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(analyze_stock, STOCK_LIST)

if __name__ == "__main__":
    logging.info("Memulai analisis multi-saham...")
    analyze_multiple_stocks()
    logging.info("Analisis selesai.")
