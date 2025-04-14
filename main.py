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

# --- [LOGGING] ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading.log"), logging.StreamHandler()]
)

# --- [ENV VARIABLES & PATHS] ---
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
RETRAIN_INTERVAL = 7

# --- [STOCK LIST] ---
# Daftar saham Anda di sini
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

# --- [UTILITY FUNCTIONS] ---
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.error("Telegram token atau chat_id tidak ditemukan. Pastikan environment variable TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID sudah di-set.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, data=data)
        if response.status_code == 200:
            logging.info("✅ Pesan berhasil dikirim ke Telegram.")
        else:
            logging.error(f"❌ Gagal mengirim pesan ke Telegram. Status code: {response.status_code}")
            logging.error(f"Response: {response.text}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="90d", interval="1h")
        if data is not None and not data.empty and len(data) >= 33:
            data["ticker"] = ticker
            return data
        if "Close" not in data.columns or data["Close"].dropna().empty:
            logging.warning(f"Tidak ada data penutupan untuk {ticker}")
            return None
        else:
            logging.warning(f"Data kosong atau kurang: {ticker}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

def calculate_indicators(df):
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

    macd = trend.MACD(df["Close"], window_slow=13, window_fast=5, window_sign=5)
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    bb = volatility.BollingerBands(df["Close"], window=20)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()

    df["Support"] = df["Low"].rolling(window=48).min()
    df["Resistance"] = df["High"].rolling(window=48).max()

    stoch = momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14)
    df["%K"] = stoch.stoch()
    df["%D"] = stoch.stoch_signal()

    df["RSI"] = momentum.RSIIndicator(df["Close"], window=14).rsi()

    df["SMA_20"] = trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["SMA_50"] = trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    df["SMA_100"] = trend.SMAIndicator(df["Close"], window=100).sma_indicator()

    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()

    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=10).adx()

    # Label prediksi harga besok (misalnya 6 jam dari sekarang)
    df["future_high"] = df["High"].shift(-6)
    df["future_low"] = df["Low"].shift(-6)

    return df.dropna()

# --- [MODEL TRAINING] ---
def tune_lightgbm_regressor(X, y):
    param_grid = {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 50]
    }
    model = lgb.LGBMRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
    grid_search.fit(X, y)
    logging.info(f"Best params (regressor): {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    return model

def train_classifier(X, y_binary):
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight='balanced')
    model.fit(X, y_binary)
    return model

def train_lstm(X, y):
    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_lstm, y, 
              epochs=50, 
              batch_size=32, 
              validation_split=0.2, 
              callbacks=[early_stop],
              verbose=1)
            
    retrain = not all(os.path.exists(path) for path in [model_high_path, model_low_path, model_cls_path, model_lstm_path, "scaler_target.pkl"])
    # Simpan scaler
    with open('scaler_target.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return model

try:
    with open('scaler_target.pkl', 'rb') as f:
        scaler_target = pickle.load(f)
except FileNotFoundError:
    logging.error("File scaler_target.pkl tidak ditemukan.")
    scaler_target = None

# --- [MAIN ANALYSIS FUNCTION] ---
def analyze_stock(ticker):
    try:
        df = get_stock_data(ticker)

        if df is None or df.empty:
            logging.warning(f"Data kosong untuk {ticker}, lewati analisis.")
            return None

        df = calculate_indicators(df)

        # List of features used for model training
        features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_20", "SMA_50", "SMA_100", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
        df = df.dropna(subset=features + ["future_high", "future_low"])

        X = df[features]  # Ensure X contains the exact same features as used during training
        y_high = df["future_high"]
        y_low = df["future_low"]
        y_binary = (df["future_high"] > df["Close"]).astype(int)

        # Split data for training
        X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2)
        X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2, random_state=42)
        y_train_low = y_low[X_train.index]
        y_train_cls = y_binary[X_train.index]

        retrain = True
        if os.path.exists(model_high_path) and os.path.exists(model_low_path) and os.path.exists(model_cls_path) and os.path.exists(model_lstm_path):
            last_modified = datetime.fromtimestamp(os.path.getmtime(model_high_path))
            if (datetime.now() - last_modified).days < RETRAIN_INTERVAL:
                retrain = False

        if retrain:
            # Retrain models if necessary
            logging.info(f"Melakukan retraining model untuk {ticker}...")

            # Train models
            model_high = train_lightgbm(X_train, y_train_high)
            joblib.dump(model_high, model_high_path)

            model_low = train_lightgbm(X_train, y_train_low)
            joblib.dump(model_low, model_low_path)

            model_cls = train_classifier(X_train, y_train_cls)
            joblib.dump(model_cls, model_cls_path)

            model_lstm = train_lstm(X_train, y_train_high)
            model_lstm.save(model_lstm_path)

            logging.info(f"Model retrained untuk {ticker}")
        else:
            logging.info(f"Model sudah up-to-date untuk {ticker}. Tidak perlu retraining.")

        # Load models
        model_high = joblib.load(model_high_path)
        model_low = joblib.load(model_low_path)
        model_cls = joblib.load(model_cls_path)
        model_lstm = load_model(model_lstm_path)

        # Predict using models
        predicted_high = model_high.predict(X[features])  # Ensure passing the correct columns
        predicted_low = model_low.predict(X[features])
        predicted_class = model_cls.predict(X[features])

        # For LSTM model, rescale and reshape input
        X_lstm_scaled = scaler_target.transform(X[features])
        X_lstm_scaled = np.reshape(X_lstm_scaled, (X_lstm_scaled.shape[0], 1, X_lstm_scaled.shape[1]))
        predicted_lstm = model_lstm.predict(X_lstm_scaled)

        # Store predictions
        df["predicted_high"] = predicted_high
        df["predicted_low"] = predicted_low
        df["predicted_class"] = predicted_class
        df["predicted_lstm"] = predicted_lstm

        # Log evaluation metrics
        evaluation = {
            "ticker": ticker,
            "mae_high": mean_absolute_error(y_high, predicted_high),
            "mae_low": mean_absolute_error(y_low, predicted_low),
            "mape_high": mean_absolute_percentage_error(y_high, predicted_high),
            "mape_low": mean_absolute_percentage_error(y_low, predicted_low)
        }

        with open(EVALUATION_LOG_PATH, "a") as eval_file:
            eval_file.write(f"{evaluation}\n")

        # Send message to Telegram
        message = f"Prediksi untuk {ticker}:\n"
        message += f"Prediksi Harga Tinggi: {predicted_high[-1]}\n"
        message += f"Prediksi Harga Rendah: {predicted_low[-1]}\n"
        message += f"Prediksi Klasifikasi: {'Naik' if predicted_class[-1] == 1 else 'Turun'}\n"
        message += f"Prediksi LSTM: {predicted_lstm[-1]}\n"

        send_telegram_message(message)

        # Save results to CSV
        df.to_csv(PREDICTION_LOG_PATH, mode='a', header=not os.path.exists(PREDICTION_LOG_PATH))

    except Exception as e:
        logging.error(f"Terjadi kesalahan saat menganalisis {ticker}: {e}")

        # Simpan data dalam file CSV untuk setiap saham
        df.to_csv(PREDICTION_LOG_PATH, mode='a', header=not os.path.exists(PREDICTION_LOG_PATH))

    except Exception as e:
        logging.error(f"Terjadi kesalahan saat menganalisis {ticker}: {e}")

# --- [MULTI-THREADING ANALYSIS] ---
def analyze_multiple_stocks():
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(analyze_stock, STOCK_LIST)

# --- [JALANKAN ANALISIS SEKALI SAJA] ---
if __name__ == "__main__":
    try:
        logging.info("Memulai analisis saham...")
        analyze_multiple_stocks()  # Jalankan analisis untuk semua saham
        logging.info("Analisis selesai.")
    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")
