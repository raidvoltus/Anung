import os, time, joblib, requests, logging
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf
from ta import momentum, trend, volatility, volume
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

# === Konfigurasi Bot ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
ATR_MULTIPLIER = 2.5
RETRAIN_INTERVAL = 1
MODEL_HIGH_PATH = "model_high.pkl"
MODEL_LOW_PATH = "model_low.pkl"
MODEL_LSTM_PATH = "model_lstm.h5"
BACKUP_CSV_PATH = "stock_data_backup.csv"

# === Daftar Saham ===
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
]

# === Logging ===
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler = RotatingFileHandler("trading.log", maxBytes=5*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

# === Kirim Pesan Telegram ===
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# === Ambil & Validasi Data ===
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="60d", interval="30m")
        if data is not None and not data.empty and len(data) >= 200:
            data["ticker"] = ticker
            return data
        else:
            logging.warning(f"Data kosong atau kurang: {ticker}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

# === Hitung Indikator Teknikal ===
def calculate_indicators(df):
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=10).average_true_range()
    macd = trend.MACD(df["Close"], window_slow=13, window_fast=5, window_sign=5)
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    bb = volatility.BollingerBands(df["Close"], window=12)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["Support"] = df["Low"].rolling(window=24).min()
    df["Resistance"] = df["High"].rolling(window=24).max()
    stoch = momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=10)
    df["%K"] = stoch.stoch()
    df["%D"] = stoch.stoch_signal()
    df["RSI"] = momentum.RSIIndicator(df["Close"], window=10).rsi()
    df["SMA_50"] = trend.SMAIndicator(df["Close"], window=24).sma_indicator()
    df["SMA_200"] = trend.SMAIndicator(df["Close"], window=48).sma_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=10).adx()
    df["future_high"] = df["High"].shift(-1)
    df["future_low"] = df["Low"].shift(-1)
    return df.dropna()

# === Training Model LightGBM ===
def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    return model

# === Training Model LSTM ===
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
    model.fit(X, y, epochs=55, batch_size=32, verbose=1)
    return model

# === Proses Analisis Saham ===
def analyze_stock(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None
    df = calculate_indicators(df)

    features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
    df = df.dropna(subset=features + ["future_high", "future_low"])
    X = df[features]
    y_high, y_low = df["future_high"], df["future_low"]

    if should_retrain():
        X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2)
        _, _, y_train_low, _ = train_test_split(X, y_low, test_size=0.2)

        model_high = train_lightgbm(X_train, y_train_high)
        model_low = train_lightgbm(X_train, y_train_low)
        model_lstm = train_lstm(X_train, y_train_high)

        joblib.dump(model_high, MODEL_HIGH_PATH)
        joblib.dump(model_low, MODEL_LOW_PATH)
        model_lstm.save(MODEL_LSTM_PATH)
        update_last_train_time()
    else:
        model_high = joblib.load(MODEL_HIGH_PATH)
        model_low = joblib.load(MODEL_LOW_PATH)
        model_lstm = load_model(MODEL_LSTM_PATH)

    X_latest = X.iloc[-1:].values
    pred_high = model_high.predict(X_latest)[0]
    pred_low = model_low.predict(X_latest)[0]

    # Estimasi probabilitas naik
    high_pred_class = model_high.predict_proba(X_latest)[0] if hasattr(model_high, "predict_proba") else None
    probability = high_pred_class[1] if high_pred_class is not None and len(high_pred_class) > 1 else 0.5

    if probability < MIN_PROBABILITY:
        return None

    current_price = df["Close"].iloc[-1]
    action = "beli" if pred_high > current_price else "jual"

    return {
        "ticker": ticker,
        "harga": round(current_price, 2),
        "take_profit": round(pred_high, 2),
        "stop_loss": round(pred_low, 2),
        "aksi": action,
        "probabilitas": round(probability, 2)
    }

# === Eksekusi & Kirim Sinyal ===
if __name__ == "__main__":
    logging.info("🚀 Memulai analisis saham...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))
    results = [r for r in results if r]

    top_5 = sorted(results, key=lambda x: x["take_profit"], reverse=True)[:5]
    if top_5:
        message = "<b>📊 Top 5 Sinyal Trading Hari Ini (Prob > {:.0%}):</b>\n".format(MIN_PROBABILITY)
        for r in top_5:
            message += (f"\n🔹 {r['ticker']}\n   💰 Harga: {r['harga']:.2f}\n   "
                        f"🎯 TP: {r['take_profit']:.2f}\n   🛑 SL: {r['stop_loss']:.2f}\n   "
                        f"📈 Prob: {r['probabilitas']:.2%}\n   📌 Aksi: <b>{r['aksi'].upper()}</b>\n")
        send_telegram_message(message)
    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Selesai dan data disimpan.")
