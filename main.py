import os, time, joblib, requests, logging
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
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading.log"), logging.StreamHandler()]
)

# === ENV VARIABLES ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# === PATHS ===
model_cls_path = "model_cls.txt"
model_high_path = "model_high.txt"
model_low_path = "model_low.txt"
model_lstm_path = "model_lstm.h5"
BACKUP_CSV_PATH = "stock_data_backup.csv"
ATR_MULTIPLIER = 2.5
RETRAIN_INTERVAL = 7

# === STOCK LIST ===
STOCK_LIST = ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]  # edit sesuai kebutuhan

# === UTILITY ===
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("Telegram token/chat_id belum diset.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, data=data)
        if r.status_code == 200:
            logging.info("✅ Pesan dikirim ke Telegram.")
        else:
            logging.error(f"❌ Gagal kirim pesan: {r.status_code} {r.text}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def get_stock_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="60d", interval="1h")
        if df is not None and not df.empty and len(df) >= 33:
            df["ticker"] = ticker
            return df
        else:
            logging.warning(f"Data kosong/kurang: {ticker}")
    except Exception as e:
        logging.error(f"Error ambil data {ticker}: {e}")
    return None

def calculate_indicators(df):
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=10).average_true_range()
    macd = trend.MACD(df["Close"], window_slow=21, window_fast=9, window_sign=5)
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
    df["SMA_9"] = trend.SMAIndicator(df["Close"], window=8).sma_indicator()
    df["SMA_20"] = trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=10).adx()
    df["future_high"] = df["High"].shift(-1)
    df["future_low"] = df["Low"].shift(-1)
    return df.dropna()

# === MODEL TRAINING ===
def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    return model

def train_classifier(X, y):
    model = lgb.LGBMClassifier(n_estimators=33, learning_rate=0.05, class_weight="balanced")
    model.fit(X, y)
    return model

def train_lstm(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_lstm, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
    return model

# === ANALISIS SAHAM ===
def analyze_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df is None:
            return None

        df = calculate_indicators(df)
        features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_9", "SMA_20", "SMA_50", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
        df = df.dropna(subset=features + ["future_high", "future_low"])
        X = df[features]
        y_high = df["future_high"]
        y_low = df["future_low"]
        y_binary = (df["future_high"] > df["Close"]).astype(int)

        X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2)
        _, _, y_train_low, _ = train_test_split(X, y_low, test_size=0.2)
        X_train_cls, _, y_train_cls, _ = train_test_split(X, y_binary, test_size=0.2)

        retrain = True
        if all(os.path.exists(p) for p in [model_high_path, model_low_path, model_cls_path, model_lstm_path]):
            last_mod = datetime.fromtimestamp(os.path.getmtime(model_high_path))
            if (datetime.now() - last_mod).days < RETRAIN_INTERVAL:
                retrain = False

        if retrain:
            model_high = train_lightgbm(X_train, y_train_high)
            model_low = train_lightgbm(X_train, y_train_low)
            model_cls = train_classifier(X_train_cls, y_train_cls)
            model_lstm = train_lstm(X_train, y_train_high)
            joblib.dump(model_high, model_high_path)
            joblib.dump(model_low, model_low_path)
            joblib.dump(model_cls, model_cls_path)
            model_lstm.save(model_lstm_path)
        else:
            model_high = joblib.load(model_high_path)
            model_low = joblib.load(model_low_path)
            model_cls = joblib.load(model_cls_path)
            model_lstm = tf.keras.models.load_model(model_lstm_path)
            model_lstm.compile(optimizer="adam", loss="mean_squared_error")

        current_price = df["Close"].iloc[-1]
        
        # Pastikan input model adalah array 2D
        X_input = X.iloc[-1:].values.reshape(1, -1)

        # Prediksi
        pred_high = model_high.predict(X_input)[0]
        pred_low = model_low.predict(X_input)[0]
        prob_up = model_cls.predict_proba(X_input)[0][1]

        # Cek validitas sinyal prediksi
        if prob_up < 0.75 or pred_high <= current_price or pred_low >= current_price:
            return None

        # Menghitung profit potential dan mengembalikan hasil analisis
        return {
            "ticker": ticker,
            "harga": current_price,
            "take_profit": pred_high,
            "stop_loss": pred_low,
            "aksi": "buy",
            "profit_pct": round((pred_high - current_price) / current_price * 100, 2),
            "probability": prob_up
        }

    except Exception as e:
        logging.error(f"Error analisis {ticker}: {e}")
        return None

def plot_probability_distribution(results):
    probs = [r["probability"] for r in results if r]
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(0.75, color='red', linestyle='--', label='Threshold 0.75')
    plt.title("Distribusi Probabilitas Kenaikan Saham")
    plt.xlabel("Probabilitas Naik")
    plt.ylabel("Jumlah Saham")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("probability_distribution.png")
    plt.close()

# === MAIN ===
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
