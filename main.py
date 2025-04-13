main.py

# === Instalasi Dependensi (Hanya Perlu Sekali) ===
!pip install yfinance lightgbm tensorflow joblib requests numpy pandas ta

# === Import Library ===
import os
import logging
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf
import joblib
import requests
import ta
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import RotatingFileHandler
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# === Konfigurasi Telegram & Variabel ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
ATR_MULTIPLIER = 2.5
RETRAIN_INTERVAL = 7
STOCK_LIST = [
    "ACES.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", "ANTM.JK",
    "ARTO.JK", "ASII.JK", "AUTO.JK", "AVIA.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK",
    "BBTN.JK", "BBYB.JK", "BDKR.JK", "BFIN.JK", "BMRI.JK", "BMTR.JK", "BNGA.JK",
    "BRIS.JK", "BRMS.JK", "BRPT.JK", "BSDE.JK", "BTPS.JK", "CMRY.JK", "CPIN.JK",
    "CTRA.JK", "DEWA.JK", "DSNG.JK", "ELSA.JK", "EMTK.JK", "ENRG.JK", "ERAA.JK",
    "ESSA.JK", "EXCL.JK", "FILM.JK", "GGRM.JK", "GJTL.JK", "GOTO.JK", "HEAL.JK",
    "HMSP.JK", "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INDY.JK", "INKP.JK",
    "INTP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK", "MDKA.JK",
    "MEDC.JK", "MIKA.JK", "MNCN.JK", "MTEL.JK", "MYOR.JK", "NCKL.JK", "PGAS.JK",
    "PNLF.JK", "PTBA.JK", "PTPP.JK", "PWON.JK", "ROTI.JK", "SAME.JK", "SCMA.JK",
    "SIDO.JK", "SILO.JK", "SMGR.JK", "SMRA.JK", "TBIG.JK", "TINS.JK", "TKIM.JK",
    "TLKM.JK", "TOWR.JK", "TPIA.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSKT.JK",
    "WTON.JK"
] # Contoh daftar saham


MODEL_HIGH_PATH = "model_high.pkl"
MODEL_LOW_PATH = "model_low.pkl"
MODEL_LSTM_PATH = "model_lstm.h5"
BACKUP_CSV_PATH = "stock_data_backup.csv"

# === Logging ===
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler = RotatingFileHandler("trading.log", maxBytes=5*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

# === Fungsi Kirim Telegram ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# === Validasi Data ===
def validate_stock_data(df, ticker):
    if df is None or df.empty or len(df) < 200:
        logging.warning(f"📉 Data terlalu sedikit atau kosong untuk {ticker}")
        return False
    return True

# === Ambil Data Saham ===
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="60d", interval="30m")
        if validate_stock_data(data, ticker):
            data["ticker"] = ticker
            return data
    except Exception as e:
        logging.error(f"❌ Error mengambil data {ticker}: {e}")
    return None

# === Hitung Indikator Teknikal (Revisi dengan fungsi ta) ===
def calculate_indicators(df):
    # ATR: 14 (standar), dikurangi jadi 10 untuk lebih cepat respons
    atr_indicator = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=10)
    df["ATR"] = atr_indicator.average_true_range()

    # MACD: default tetap (12,26,9), masih oke untuk intraday
    macd_indicator = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd_indicator.macd()
    df["Signal_Line"] = macd_indicator.macd_signal()
    df["MACD_Hist"] = macd_indicator.macd_diff()

    # Bollinger Bands: ubah window dari 20 ke 12 (karena 12 candle = 1 hari)
    bb_indicator = ta.volatility.BollingerBands(df["Close"], window=12, window_dev=2)
    df["BB_Upper"] = bb_indicator.bollinger_hband()
    df["BB_Lower"] = bb_indicator.bollinger_lband()

    # Support & Resistance: ubah dari 50 ke 24 (±2 hari bursa)
    df["Support"] = df["Low"].rolling(window=24).min()
    df["Resistance"] = df["High"].rolling(window=24).max()

    # Stochastic Oscillator: ubah dari 14 ke 10
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=10, smooth_window=3)
    df["%K"] = stoch.stoch()
    df["%D"] = stoch.stoch_signal()

    # RSI: ubah dari 14 ke 10
    rsi_indicator = ta.momentum.RSIIndicator(df["Close"], window=10)
    df["RSI"] = rsi_indicator.rsi()

    # SMA: ubah dari 50/200 ke 24/48 (2 hari dan 4 hari)
    df["SMA_50"] = ta.trend.SMAIndicator(df["Close"], window=24).sma_indicator()
    df["SMA_200"] = ta.trend.SMAIndicator(df["Close"], window=48).sma_indicator()

    # VWAP: tetap karena akumulasi volume harian, tetap relevan
    vwap_indicator = ta.volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"])
    df["VWAP"] = vwap_indicator.volume_weighted_average_price()

    # ADX: ubah window dari 14 ke 10
    adx_indicator = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=10)
    df["ADX"] = adx_indicator.adx()

    # Target: future high/low next candle (30 menit ke depan)
    df["future_high"] = df["High"].shift(-1)
    df["future_low"] = df["Low"].shift(-1)

    df.dropna(inplace=True)
    return df

# === Training LightGBM ===
def train_lightgbm(X_train, y_train):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31)
    model.fit(X_train, y_train)
    return model

# === Training LSTM ===
def train_lstm(X_train, y_train):
    X_lstm = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_lstm, y_train, epochs=55, batch_size=32, verbose=1)
    return model

# === Analisis Saham & Prediksi ===
def analyze_stock(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None

    df = calculate_indicators(df)
    if df.empty:
        return None

    # Gunakan fitur indikator yang dioptimalkan
    features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
    df = df.dropna(subset=features + ["future_high", "future_low"])

    X = df[features]
    y_high, y_low = df["future_high"], df["future_low"]

    X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2, random_state=42)
    X_train, _, y_train_low, _ = train_test_split(X, y_low, test_size=0.2, random_state=42)

    model_high = train_lightgbm(X_train, y_train_high)
    model_low = train_lightgbm(X_train, y_train_low)
    model_lstm = train_lstm(X_train, y_train_high)

    # Simpan model (opsional)
    joblib.dump(model_high, MODEL_HIGH_PATH)
    joblib.dump(model_low, MODEL_LOW_PATH)
    model_lstm.save(MODEL_LSTM_PATH)

    pred_high = model_high.predict(X.iloc[-1:].values)[0]
    pred_low = model_low.predict(X.iloc[-1:].values)[0]
    current_price = df["Close"].iloc[-1]
    action = "beli" if pred_high > current_price else "jual"

    # Hitung risk-reward ratio
    risk = current_price - pred_low
    reward = pred_high - current_price
    if risk <= 0 or reward / risk < 3:
        return None  # Abaikan sinyal jika risk-reward tidak memenuhi 1:3

    return {
        "ticker": ticker,
        "harga": round(current_price, 2),
        "take_profit": round(pred_high, 2),
        "stop_loss": round(pred_low, 2),
        "aksi": action
    }

# === Eksekusi Multiprocessing & Pengiriman Sinyal Top 5 dalam Satu Pesan ===
if __name__ == "__main__":
    logging.info("🚀 Memulai analisis saham...")

    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    # Filter hasil yang valid
    results = [r for r in results if r is not None]

    # Pilih top 5 saham (misalnya berdasarkan take_profit tertinggi)
    top_5 = sorted(results, key=lambda x: x["take_profit"], reverse=True)[:5]

    if top_5:
        message = "<b>📊 Top 5 Sinyal Trading Hari Ini:</b>\n"
        for r in top_5:
            message += (f"\n🔹 {r['ticker']}\n   💰 Harga: {r['harga']:.2f}\n   "
                        f"🎯 TP: {r['take_profit']:.2f}\n   🛑 SL: {r['stop_loss']:.2f}\n   "
                        f"📌 Aksi: <b>{r['aksi'].upper()}</b>\n")
        send_telegram_message(message)

    # Simpan hasil ke CSV backup
    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Bot selesai & berhenti total.")
    exit()
