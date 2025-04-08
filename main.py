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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from datetime import datetime

# === Konfigurasi dari Environment (GitHub Secrets) ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
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

# === Path Model dan Backup ===
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_HIGH_PATH = os.path.join(DATA_DIR, "model_high.pkl")
MODEL_LOW_PATH = os.path.join(DATA_DIR, "model_low.pkl")
MODEL_LSTM_PATH = os.path.join(DATA_DIR, "model_lstm.h5")
BACKUP_CSV_PATH = os.path.join(DATA_DIR, "stock_data_backup.csv")

# === Logging ===
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler = RotatingFileHandler(os.path.join(DATA_DIR, "trading.log"), maxBytes=5*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

# === Fungsi Telegram ===
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("Telegram token atau chat ID belum diset.")
        return
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
        data = stock.history(period="6mo")
        if validate_stock_data(data, ticker):
            data["ticker"] = ticker
            return data
    except Exception as e:
        logging.error(f"❌ Error mengambil data {ticker}: {e}")
    return None

# === Indikator Teknikal ===
def calculate_indicators(df):
    atr_indicator = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ATR"] = atr_indicator.average_true_range()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["Support"] = df["Low"].rolling(window=50).min()
    df["Resistance"] = df["High"].rolling(window=50).max()
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["%K"] = stoch.stoch()
    df["%D"] = stoch.stoch_signal()
    rsi = ta.momentum.RSIIndicator(df["Close"])
    df["RSI"] = rsi.rsi()
    df["SMA_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    df["SMA_200"] = ta.trend.SMAIndicator(df["Close"], window=200).sma_indicator()
    vwap = ta.volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"])
    df["VWAP"] = vwap.volume_weighted_average_price()
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx.adx()
    df["future_high"] = df["High"].shift(-1)
    df["future_low"] = df["Low"].shift(-1)
    df.dropna(inplace=True)
    return df

# === Training LightGBM ===
def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31)
    model.fit(X, y)
    return model

# === Training LSTM ===
def train_lstm(X, y):
    X_lstm = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_lstm, y, epochs=55, batch_size=32, verbose=0)
    return model

# === Analisa Saham ===
def analyze_stock(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None
    df = calculate_indicators(df)
    features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
    df.dropna(subset=features + ["future_high", "future_low"], inplace=True)
    if df.empty:
        return None
    X = df[features]
    y_high, y_low = df["future_high"], df["future_low"]
    X_train, _, y_high_train, _ = train_test_split(X, y_high, test_size=0.2)
    model_high = train_lightgbm(X_train, y_high_train)
    model_low = train_lightgbm(X_train, y_low)
    model_lstm = train_lstm(X_train, y_high_train)
    joblib.dump(model_high, MODEL_HIGH_PATH)
    joblib.dump(model_low, MODEL_LOW_PATH)
    model_lstm.save(MODEL_LSTM_PATH)
    pred_high = model_high.predict([X.iloc[-1]])[0]
    pred_low = model_low.predict([X.iloc[-1]])[0]
    current_price = df["Close"].iloc[-1]
    action = "beli" if pred_high > current_price else "jual"
    risk = current_price - pred_low
    reward = pred_high - current_price
    if risk <= 0 or reward / risk < 3:
        return None
    return {
        "ticker": ticker,
        "harga": round(current_price, 2),
        "take_profit": round(pred_high, 2),
        "stop_loss": round(pred_low, 2),
        "aksi": action
    }

# === Main Execution ===
if __name__ == "__main__":
    logging.info("🚀 Memulai analisis saham...")
    all_results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for result in executor.map(analyze_stock, STOCK_LIST):
            if result:
                logging.info(f"✅ {result['ticker']} lolos filter! TP: {result['take_profit']}, SL: {result['stop_loss']}")
            else:
                logging.info(f"❌ Tidak ada sinyal valid.")
            all_results.append(result)

    results = [r for r in all_results if r is not None]
    top_5 = sorted(results, key=lambda x: x["take_profit"], reverse=True)[:5]

    if top_5:
        message = "<b>📊 Top 5 Sinyal Trading Hari Ini:</b>\n"
        for r in top_5:
            message += (f"\n🔹 {r['ticker']}\n   💰 Harga: {r['harga']:.2f}\n   "
                        f"🎯 TP: {r['take_profit']:.2f}\n   🛑 SL: {r['stop_loss']:.2f}\n   "
                        f"📌 Aksi: <b>{r['aksi'].upper()}</b>\n")
        send_telegram_message(message)
        logging.info("📨 Sinyal dikirim ke Telegram!")
    else:
        logging.info("⚠️ Tidak ada sinyal yang memenuhi syarat hari ini.")

    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Bot selesai.")
