# === Import Library ===
import os
import time
import threading
import joblib
import requests
import logging
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
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
ATR_MULTIPLIER   = 2.5
RETRAIN_INTERVAL = 7
BACKUP_CSV_PATH  = "stock_data_backup.csv"

# === Daftar Saham ===
STOCK_LIST = [
    "ACES.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "AUTO.JK",
    "AVIA.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BBYB.JK", "BDKR.JK", "BFIN.JK", "BMRI.JK", "BMTR.JK",
    "BNGA.JK", "BRIS.JK", "BRMS.JK", "BRPT.JK", "BSDE.JK", "BTPS.JK", "CMRY.JK", "CPIN.JK", "CTRA.JK", "DEWA.JK",
    "DSNG.JK", "ELSA.JK", "EMTK.JK", "ENRG.JK", "ERAA.JK", "ESSA.JK", "EXCL.JK", "FILM.JK", "GGRM.JK", "GJTL.JK",
    "GOTO.JK", "HEAL.JK", "HMSP.JK", "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INDY.JK", "INET.JK", "INKP.JK",
    "INTP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KIJA.JK", "KLBF.JK", "KPIG.JK", "LSIP.JK", "MAPA.JK",
    "MAPI.JK", "MARK.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK", "MIDI.JK", "MIKA.JK", "MNCN.JK", "MTEL.JK", "MYOR.JK",
    "NCKL.JK", "NISP.JK", "PANI.JK", "PGAS.JK", "PGEO.JK", "PNLF.JK", "PTBA.JK", "PTPP.JK", "PTRO.JK", "PWON.JK",
    "RAJA.JK", "SCMA.JK", "SIDO.JK", "SMGR.JK", "SMIL.JK", "SMRA.JK", "SRTG.JK", "SSIA.JK", "SSMS.JK", "SURI.JK",
    "TINS.JK", "TKIM.JK", "TLKM.JK", "TOBA.JK", "TOWR.JK", "TPIA.JK", "TRIN.JK", "TSPC.JK", "UNTR.JK", "UNVR.JK"
]

# === Logging Setup ===
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler   = RotatingFileHandler("trading.log", maxBytes=5*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

# === Lock untuk Thread-Safe Model Saving ===
model_save_lock = threading.Lock()

# === Fungsi Kirim Telegram ===
def send_telegram_message(message: str):
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# === Ambil & Validasi Data Saham ===
def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period="1y", interval="1d")
        if df is not None and not df.empty and len(df) >= 200:
            df["ticker"] = ticker
            return df
        logging.warning(f"Data kosong/kurang untuk {ticker}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

# === Hitung Indikator Teknikal ===
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ATR"]        = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    macd             = trend.MACD(df["Close"])
    df["MACD"]       = macd.macd()
    df["MACD_Hist"]  = macd.macd_diff()
    bb               = volatility.BollingerBands(df["Close"], window=20)
    df["BB_Upper"]   = bb.bollinger_hband()
    df["BB_Lower"]   = bb.bollinger_lband()
    df["Support"]    = df["Low"].rolling(window=20).min()
    df["Resistance"] = df["High"].rolling(window=20).max()
    stoch            = momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14)
    df["RSI"]        = momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["SMA_20"]     = trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["SMA_50"]     = trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    df["SMA_200"]    = trend.SMAIndicator(df["Close"], window=200).sma_indicator()
    df["VWAP"]       = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"]        = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    df["future_high"] = df["High"].shift(-1)
    df["future_low"]  = df["Low"].shift(-1)
    return df.dropna()

# === Training LightGBM ===
def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    return model

# === Training LSTM ===
def train_lstm(X: pd.DataFrame, y: pd.Series) -> Sequential:
    X_arr = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_arr, y, epochs=55, batch_size=32, verbose=0)
    return model

# === Hitung Probabilitas Arah Prediksi ===
def calculate_probability(model, X: pd.DataFrame, y_true: pd.Series) -> float:
    y_pred = model.predict(X)
    y_pred_series = pd.Series(y_pred, index=X.index)
    close_price   = X["Close"]
    correct_dir   = ((y_pred_series > close_price) & (y_true > close_price)) | \
                    ((y_pred_series < close_price) & (y_true < close_price))
    return correct_dir.sum() / len(correct_dir)

# === Fungsi Utama Per-Ticker ===
def analyze_stock(ticker: str):
    df = get_stock_data(ticker)
    if df is None:
        return None

    df = calculate_indicators(df)

    # -- Early filter: harga, volume, volatilitas --
    price      = df["Close"].iloc[-1]
    avg_volume = df["Volume"].tail(20).mean()
    atr        = df["ATR"].iloc[-1]

    if price < 50:
        logging.info(f"{ticker} dilewati: harga terlalu rendah ({price:.2f})")
        return None
    if avg_volume < 10000:
        logging.info(f"{ticker} dilewati: volume terlalu rendah ({avg_volume:.0f})")
        return None
    if (atr / price) < 0.005:
        logging.info(f"{ticker} dilewati: volatilitas terlalu rendah (ATR={atr:.4f})")
        return None

    # -- Siapkan fitur & label --
    features = ["Close","ATR","RSI","MACD","MACD_Hist","SMA_20","SMA_50",
                "SMA_200","BB_Upper","BB_Lower","Support","Resistance","VWAP","ADX"]
    df = df.dropna(subset=features + ["future_high","future_low"])
    X      = df[features]
    y_high = df["future_high"]
    y_low  = df["future_low"]

    # -- Split data sinkron --
    X_tr, X_te, yh_tr, yh_te, yl_tr, yl_te = train_test_split(
        X, y_high, y_low, test_size=0.2, random_state=42
    )

    # -- Load atau Train LightGBM High --
    high_path = f"model_high_{ticker}.pkl"
    if os.path.exists(high_path):
        model_high = joblib.load(high_path)
        logging.info(f"Loaded LGB High for {ticker}")
    else:
        model_high = train_lightgbm(X_tr, yh_tr)
        with model_save_lock:
            joblib.dump(model_high, high_path)
        logging.info(f"Trained & saved LGB High for {ticker}")

    # -- Load atau Train LightGBM Low --
    low_path = f"model_low_{ticker}.pkl"
    if os.path.exists(low_path):
        model_low = joblib.load(low_path)
        logging.info(f"Loaded LGB Low for {ticker}")
    else:
        model_low = train_lightgbm(X_tr, yl_tr)
        with model_save_lock:
            joblib.dump(model_low, low_path)
        logging.info(f"Trained & saved LGB Low for {ticker}")

    # -- Load atau Train LSTM --
    lstm_path = f"model_lstm_{ticker}.keras"
    if os.path.exists(lstm_path):
        model_lstm = load_model(lstm_path)
        logging.info(f"Loaded LSTM for {ticker}")
    else:
        model_lstm = train_lstm(X_tr, yh_tr)
        with model_save_lock:
            model_lstm.save(lstm_path)
        logging.info(f"Trained & saved LSTM for {ticker}")

    # -- Hitung probabilitas --
    prob_high = calculate_probability(model_high, X_te, yh_te)
    prob_low  = calculate_probability(model_low,  X_te, yl_te)

    if prob_high < 0.6 or prob_low < 0.6:
        logging.info(f"{ticker} dilewati: prob rendah (H={prob_high:.2f}, L={prob_low:.2f})")
        return None

    # -- Prediksi sinyal terbaru --
    X_last    = X.iloc[-1:]
    ph        = model_high.predict(X_last)[0]
    pl        = model_low.predict(X_last)[0]
    action    = "beli" if ph > price else "jual"
    prob_succ = (prob_high + prob_low) / 2

    return {
        "ticker":       ticker,
        "harga":        round(price,     2),
        "take_profit":  round(ph,        2),
        "stop_loss":    round(pl,        2),
        "aksi":         action,
        "prob_high":    round(prob_high, 2),
        "prob_low":     round(prob_low,  2),
        "prob_success": round(prob_succ,  2)
    }

# === Eksekusi & Kirim Sinyal ===
if __name__ == "__main__":
    logging.info("🚀 Memulai analisis saham...")
    max_workers = min(8, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    # Filter hasil valid
    results = [r for r in results if r]

    # Simpan backup CSV
    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Backup CSV disimpan")

    # Ambil top 5 berdasarkan take_profit
    top_5 = sorted(results, key=lambda x: x["take_profit"], reverse=True)[:5]
    if top_5:
        message = "<b>📊 Top 5 Sinyal Trading Hari Ini:</b>\n"
        for r in top_5:
            message += (
                f"\n🔹 {r['ticker']}\n"
                f"   💰 Harga: {r['harga']:.2f}\n"
                f"   🎯 TP: {r['take_profit']:.2f}\n"
                f"   🛑 SL: {r['stop_loss']:.2f}\n"
                f"   ✅ Probabilitas: {r['prob_success']*100:.1f}%\n"
                f"   📌 Aksi: <b>{r['aksi'].upper()}</b>\n"
            )
        send_telegram_message(message)

    logging.info("✅ Selesai.")
