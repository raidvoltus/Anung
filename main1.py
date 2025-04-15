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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

# --- [LOGGING] ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading.log"), logging.StreamHandler()]
)

# --- [KONFIGURASI & PATH] ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
model_cls_path = "model_cls.txt"
model_high_path = "model_high.txt"
model_low_path = "model_low.txt"
model_lstm_path = "model_lstm.h5"
BACKUP_CSV_PATH = "stock_data_backup.csv"
PREDICTION_LOG_PATH = "weekly_predictions.csv"
EVALUATION_LOG_PATH = "weekly_evaluation.txt"
ATR_MULTIPLIER = 2.5
TP_MULTIPLIER = 0.01
SL_MULTIPLIER = 0.015
MIN_PROBABILITY = 0.045
RETRAIN_INTERVAL = 3
yf.set_tz_cache_location("/tmp/py-yfinance-cache")  # Cache di lokasi baru
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

# --- [TELEGRAM] ---
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.error("Telegram token/chat_id tidak ditemukan.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, data=data)
        if response.status_code == 200:
            logging.info("✅ Pesan dikirim ke Telegram.")
        else:
            logging.error(f"❌ Gagal kirim pesan. Code: {response.status_code} | {response.text}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# --- [DATA SAHAM] ---
def get_stock_data(ticker, max_retries=5, delay=2):
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="730d", interval="1h")

            if data is None or data.empty or len(data) < 200:
                logging.warning(f"Data kosong atau kurang untuk {ticker}, percobaan ke-{attempt+1}")
                time.sleep(delay + random.uniform(0, 2))  # Tambahkan jitter
                continue

            if "Volume" not in data.columns:
                data["Volume"] = 0

            data["ticker"] = ticker
            return data

        except Exception as e:
            logging.error(f"Percobaan {attempt+1} - Error mengambil data {ticker}: {e}")
            time.sleep(delay + random.uniform(0, 2))  # Tunggu sebelum mencoba lagi

    logging.error(f"Gagal mengambil data untuk {ticker} setelah {max_retries} percobaan.")
    return None

# --- [INDIKATOR TEKNIKAL] ---
def calculate_indicators(df):
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=10).average_true_range()

    macd = trend.MACD(df["Close"], window_slow=13, window_fast=5, window_sign=5)
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    bb = volatility.BollingerBands(df["Close"], window=20)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()

    df["Support"] = df["Low"].rolling(window=24).min()
    df["Resistance"] = df["High"].rolling(window=24).max()

    stoch = momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14)
    df["%K"] = stoch.stoch()
    df["%D"] = stoch.stoch_signal()

    df["RSI"] = momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["SMA_20"] = trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["SMA_50"] = trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    df["SMA_100"] = trend.SMAIndicator(df["Close"], window=100).sma_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=10).adx()

    df["future_high"] = df["High"].shift(-6)
    df["future_low"] = df["Low"].shift(-6)
    return df.dropna()

# --- [TRAINING MODEL] ---
def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    return model

def train_classifier(X, y_binary):
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight='balanced')
    model.fit(X, y_binary)
    return model

def train_lstm(X, y):
    scaler = StandardScaler()  # Lebih cocok dibanding MinMaxScaler untuk distribusi normal
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler_lstm.save")  # Simpan sesuai fitur X

    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(1, X_scaled.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_lstm, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

    return model

try:
    scaler_target = joblib.load("scaler_lstm.save")
except FileNotFoundError:
    logging.warning("Scaler file tidak ditemukan.")
    scaler_target = None

def analyze_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df is None or df.empty:
            return None
            
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"Data kosong atau tidak cukup untuk {ticker}")
            return None
            
        df = calculate_indicators(df)

        features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_20", "SMA_50", "SMA_100",
                    "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
        df = df.dropna(subset=features + ["future_high", "future_low"])

        if len(X) < 10:
            logging.warning(f"Data untuk {ticker} terlalu sedikit untuk training.")
            return None
            
        X = df[features]
        y_high = df["future_high"]
        y_low = df["future_low"]
        y_binary = (df["future_high"] > df["Close"]).astype(int)
    
        X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2, random_state=42)
        y_train_low = y_low[X_train.index]
        y_train_cls = y_binary[X_train.index]

        retrain = False
        if not all([os.path.exists(p) for p in [model_high_path, model_low_path, model_cls_path, model_lstm_path]]):
            retrain = True
        else:
            last_modified = datetime.fromtimestamp(os.path.getmtime(model_high_path))
            retrain = (datetime.now() - last_modified).days >= RETRAIN_INTERVAL

        if retrain:
            logging.info(f"Retrain model untuk {ticker}...")
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

            predicted_high = model_high.predict(X)
            predicted_low = model_low.predict(X)
            predicted_class = model_cls.predict(X)

        if scaler_target is None:
              return None

        try:
              X_lstm_scaled = scaler_target.transform(X)
        except ValueError as e:
              logging.error(f"Scaler error pada {ticker}: {e}")
              return None

        if scaler_target.mean_.shape[0] != X.shape[1]:
              logging.error(f"Mismatch fitur: scaler expects {scaler_target.mean_.shape[0]}, got {X.shape[1]}")
              return None
        try:
            # Proses pembentukan fitur
            features = prepare_features(df)

            if features is None or features.empty:
                raise ValueError("Fitur tidak tersedia")

            X = features[-1:]  # ambil baris terakhir untuk prediksi

            # Normalisasi
            X_scaled = scaler.transform(X)

            # Prediksi
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            
            X_lstm_scaled = np.reshape(X_lstm_scaled, (X_lstm_scaled.shape[0], 1, X_lstm_scaled.shape[1]))
            predicted_lstm = model_lstm.predict(X_lstm_scaled)

            X_lstm_scaled = scaler_target.transform(X)
            X_lstm_scaled = np.reshape(X_lstm_scaled, (X_lstm_scaled.shape[0], 1, X_lstm_scaled.shape[1]))
            predicted_lstm = model_lstm.predict(X_lstm_scaled)

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

            # LOGIKA YANG DILONGGARKAN UNTUK MEMPERBANYAK SINYAL
            if aksi == "buy" and potensi >= 1.5 and prob >= 0.70:
                signal = {
                    "ticker": ticker,
                    "harga": harga,
                    "stop_loss": sl,
                    "aksi": aksi,
                    "probability": prob,
                    "profit_pct": potensi
                }
                return signal
            else:
                return None
            
    except Exception as e:
        logging.error(f"Error menganalisis {ticker}: {e}")
        return None

def plot_probability_distribution(results):
    try:
        df = pd.DataFrame(results)
        plt.figure(figsize=(10, 4))
        plt.hist(df["probability"], bins=10, color='skyblue', edgecolor='black')
        plt.title("Distribusi Probabilitas Sinyal")
        plt.xlabel("Probabilitas")
        plt.ylabel("Jumlah Saham")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("probability_distribution.png")
    except Exception as e:
        logging.warning(f"Gagal membuat grafik distribusi: {e}")

# Path ke file scaler
SCALER_PATH = 'scaler_target.pkl'

def buat_scaler_otomatis(ticker='BMRI.JK'):
    try:
        logging.info(f"Mengunduh data historis untuk {ticker}...")
        stock_data = yf.download(ticker, period='730d', interval='1h')
        if stock_data.empty:
            logging.warning(f"Tidak ada data yang diunduh untuk {ticker}. Tidak dapat membuat scaler.")
            return

        # Menggunakan harga penutupan
        closing_prices = stock_data['Close'].values.reshape(-1, 1)

        # Membuat dan melatih scaler
        scaler = StandardScaler()
        scaler.fit(closing_prices)

        # Menyimpan scaler ke file
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Scaler berhasil dibuat dan disimpan di {SCALER_PATH}")
    except Exception as e:
        logging.error(f"Gagal membuat scaler: {e}")

# Cek dan buat scaler jika belum ada
if not os.path.exists(SCALER_PATH):
    logging.warning(f"File {SCALER_PATH} tidak ditemukan.")
    buat_scaler_otomatis()
else:
    logging.info(f"File scaler ditemukan: {SCALER_PATH}")

if __name__ == "__main__":
    logging.info("Memulai analisis saham...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r]

    if results:
        plot_probability_distribution(results)
        top_5 = sorted(results, key=lambda x: x["take_profit"], reverse=True)[:5]
        msg = "<b>🧠 Sinyal Saham Terbaik Hari Ini:</b>\n"
        for r in top_5:
            msg += (
                f"\n🔹 {r['ticker']}\n   💰 Harga: {r['harga']:.2f}\n   "
                f"🎯 TP: {r['take_profit']:.2f}\n   🛑 SL: {r['stop_loss']:.2f}\n   "
                f"📌 Aksi: <b>{r['aksi'].upper()}</b>\n   📈 Potensi: {r['profit_pct']}%\n"
                f"   ✅ Probabilitas: {r['probability']*100:.2f}%\n"
            )
        send_telegram_message(msg)
    else:
        logging.info("⚠️ Tidak ada sinyal valid hari ini.")

    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info(f"Data disimpan ke {BACKUP_CSV_PATH}")
