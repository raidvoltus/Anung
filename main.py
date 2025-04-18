import os
import glob
import time
import threading
import joblib
import hashlib
import json
import requests
import random
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf

from typing import Optional, Dict, List, Tuple
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
HASH_PATH = "features_hash.json"
# Konstanta threshold (letakkan di atas fungsi analyze_stock)
MIN_PRICE = 100
MIN_VOLUME = 10000
MIN_VOLATILITY = 0.005
MIN_PROB = 0.8
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
def log_prediction(ticker: str, tanggal: str, pred_high: float, pred_low: float, harga_awal: float):
    with open("prediksi_log.csv", "a") as f:
        f.write(f"{ticker},{tanggal},{harga_awal},{pred_high},{pred_low}\n")

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
        df    = stock.history(period="6mo", interval="1h")
        if df is not None and not df.empty and len(df) >= 200:
            df["ticker"] = ticker
            return df
        logging.warning(f"Data kosong/kurang untuk {ticker}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

# === Hitung Indikator ===
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    HOURS_PER_DAY = 6  # jumlah jam trading aktif per hari

    # === Indikator teknikal utama ===
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    
    macd = trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Hist"] = macd.macd_diff()
    
    bb = volatility.BollingerBands(df["Close"], window=20)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()

    df["Support"] = df["Low"].rolling(window=48).min()
    df["Resistance"] = df["High"].rolling(window=48).max()

    df["RSI"] = momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["SMA_14"] = trend.SMAIndicator(df["Close"], window=14).sma_indicator()
    df["SMA_28"] = trend.SMAIndicator(df["Close"], window=28).sma_indicator()
    df["SMA_84"] = trend.SMAIndicator(df["Close"], window=84).sma_indicator()
    df["EMA_10"] = trend.EMAIndicator(df["Close"], window=10).ema_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    df["CCI"] = trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci()
    df["Momentum"] = momentum.ROCIndicator(df["Close"], window=12).roc()
    df["WilliamsR"] = momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14).williams_r()

    # === Fitur waktu harian ===
    df["hour"] = df.index.hour
    df["is_opening_hour"] = (df["hour"] == 9).astype(int)
    df["is_closing_hour"] = (df["hour"] == 14).astype(int)
    df["daily_avg"] = df["Close"].rolling(HOURS_PER_DAY).mean()
    df["daily_std"] = df["Close"].rolling(HOURS_PER_DAY).std()
    df["daily_range"] = df["High"].rolling(HOURS_PER_DAY).max() - df["Low"].rolling(HOURS_PER_DAY).min()

    # === Target prediksi: harga tertinggi & terendah BESOK ===
    df["future_high"] = df["High"].shift(-HOURS_PER_DAY).rolling(HOURS_PER_DAY).max()
    df["future_low"]  = df["Low"].shift(-HOURS_PER_DAY).rolling(HOURS_PER_DAY).min()

    return df.dropna()

# === Training LightGBM ===
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    early_stopping_rounds: Optional[int] = 50,
    random_state: int = 42
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    return model

# === Training LSTM ===
def train_lstm(
    X: pd.DataFrame,
    y: pd.Series,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    epochs: int = 55,
    batch_size: int = 32,
    verbose: int = 0
) -> Sequential:
    X_arr = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_arr, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# === Hitung Probabilitas Arah Prediksi ===
def calculate_probability(model, X: pd.DataFrame, y_true: pd.Series) -> float:
    if "Close" not in X.columns:
        raise ValueError("'Close' column is required in input features (X).")
    if len(X) != len(y_true):
        raise ValueError("Length of X and y_true must match.")

    y_pred = model.predict(X)
    y_pred_series = pd.Series(y_pred, index=X.index)
    close_price = X["Close"]

    correct_dir = ((y_pred_series > close_price) & (y_true > close_price)) | \
                  ((y_pred_series < close_price) & (y_true < close_price))
    correct_dir = correct_dir.dropna()

    if len(correct_dir) == 0:
        return 0.0

    return correct_dir.sum() / len(correct_dir)

# Fungsi load_or_train_model (letakkan sebelum atau setelah analyze_stock)
def load_or_train_model(path, train_func, X, y):
    if os.path.exists(path):
        model = joblib.load(path) if path.endswith(".pkl") else load_model(path)
        logging.info(f"Loaded model from {path}")
    else:
        model = train_func(X, y)
        with model_save_lock:
            if path.endswith(".pkl"):
                joblib.dump(model, path)
            else:
                model.save(path)
        logging.info(f"Trained & saved model to {path}")
    return model
    
def get_feature_hash(features: list[str]) -> str:
    features_str = ",".join(sorted(features))
    return hashlib.md5(features_str.encode()).hexdigest()

def check_and_reset_model_if_needed(ticker: str, current_features: list[str]):
    current_hash = get_feature_hash(current_features)

    try:
        with open(HASH_PATH, "r") as f:
            saved_hashes = json.load(f)
    except FileNotFoundError:
        saved_hashes = {}

    if saved_hashes.get(ticker) != current_hash:
        logging.info(f"{ticker}: Struktur fitur berubah — melakukan reset model")

        # Hapus LightGBM
        for suffix in ["high", "low"]:
            model_path = f"model_{suffix}_{ticker}.pkl"
            if os.path.exists(model_path):
                os.remove(model_path)
                logging.info(f"{ticker}: Model LightGBM '{suffix}' dihapus")

        # Hapus LSTM
        lstm_path = f"model_lstm_{ticker}.keras"
        if os.path.exists(lstm_path):
            os.remove(lstm_path)
            logging.info(f"{ticker}: Model LSTM dihapus")

        # Simpan hash baru
        saved_hashes[ticker] = current_hash
        with open(HASH_PATH, "w") as f:
            json.dump(saved_hashes, f, indent=2)

        logging.info(f"{ticker}: Model akan dilatih ulang dengan struktur fitur terbaru")
    else:
        logging.debug(f"{ticker}: Struktur fitur sama — model tidak di-reset")
        
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

    if price < MIN_PRICE:
        logging.info(f"{ticker} dilewati: harga terlalu rendah ({price:.2f})")
        return None
    if avg_volume < MIN_VOLUME:
        logging.info(f"{ticker} dilewati: volume terlalu rendah ({avg_volume:.0f})")
        return None
    if (atr / price) < MIN_VOLATILITY:
        logging.info(f"{ticker} dilewati: volatilitas terlalu rendah (ATR={atr:.4f})")
        return None

    # -- Siapkan fitur & label (disesuaikan untuk prediksi harga besok) --
    features = [
        "Close", "ATR", "RSI", "MACD", "MACD_Hist",
        "SMA_14", "SMA_28", "SMA_84", "EMA_10",
        "BB_Upper", "BB_Lower", "Support", "Resistance",
        "VWAP", "ADX", "CCI", "Momentum", "WilliamsR",
        "daily_avg", "daily_std", "daily_range",
        "is_opening_hour", "is_closing_hour"
    ]
    
    # --- Cek dan reset model bila fitur berubah ---
    check_and_reset_model_if_needed(ticker, features)

    df = df.dropna(subset=features + ["future_high", "future_low"])
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
    if action == "beli":
        profit_potential_pct = (ph - price) / price * 100
    else:
        profit_potential_pct = (price - pl) / price * 100
    # === Logging Prediksi ===
    tanggal = pd.Timestamp.now().strftime("%Y-%m-%d")
    log_prediction(ticker, tanggal, ph, pl, price)

    return {
        "ticker":       ticker,
        "harga":        round(price,     2),
        "take_profit":  round(ph,        2),
        "stop_loss":    round(pl,        2),
        "aksi":         action,
        "prob_high":    round(prob_high, 2),
        "prob_low":     round(prob_low,  2),
        "prob_success": round(prob_succ,  2),
        "profit_potential_pct": round(profit_potential_pct, 2),
    }

def retrain_if_needed(ticker: str):
    akurasi_map = evaluate_prediction_accuracy()
    akurasi = akurasi_map.get(ticker, 1.0)  # default 100%
    if akurasi < 0.90:
        logging.info(f"Akurasi model {ticker} rendah ({akurasi:.2%}), retraining...")
        df = get_stock_data(ticker)
        if df is None:
            return
        df = calculate_indicators(df)
        df = df.dropna(subset=["future_high", "future_low"])
        features = [
            "Close", "ATR", "RSI", "MACD", "MACD_Hist",
            "SMA_14", "SMA_28", "SMA_84", "EMA_10",
            "BB_Upper", "BB_Lower", "Support", "Resistance",
            "VWAP", "ADX", "CCI", "Momentum", "WilliamsR",
            "daily_avg", "daily_std", "daily_range",
            "is_opening_hour", "is_closing_hour"
        ]
        X = df[features]
        y_high = df["future_high"]
        y_low = df["future_low"]
        model_high = train_lightgbm(X, y_high)
        joblib.dump(model_high, f"model_high_{ticker}.pkl")
        model_low = train_lightgbm(X, y_low)
        joblib.dump(model_low, f"model_low_{ticker}.pkl")
        
def get_realized_price_data() -> pd.DataFrame:
    if not os.path.exists("prediksi_log.csv"):
        return pd.DataFrame()

    df_log = pd.read_csv("prediksi_log.csv")
    result = []

    for _, row in df_log.iterrows():
        ticker = row["ticker"]
        tanggal = pd.to_datetime(row["tanggal"])
        tanggal_target = tanggal + pd.Timedelta(days=5)
        df_future = yf.download(ticker, start=tanggal.strftime("%Y-%m-%d"), 
                                end=(tanggal_target + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                interval="1d", progress=False)
        if df_future.empty or len(df_future) < 5:
            continue
        actual_high = df_future["High"].max()
        actual_low  = df_future["Low"].min()
        result.append({
            "ticker": row["ticker"],
            "tanggal": row["tanggal"],
            "actual_high": actual_high,
            "actual_low": actual_low
        })

    return pd.DataFrame(result)
    
def evaluate_prediction_accuracy() -> Dict[str, float]:
    if not os.path.exists("prediksi_log.csv"):
        return {}

    # Baca log prediksi
    df_log = pd.read_csv("prediksi_log.csv", names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
    df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])

    # Ambil data realisasi dari yfinance
    df_data = get_realized_price_data()
    if df_data.empty:
        return {}

    df_data["tanggal"] = pd.to_datetime(df_data["tanggal"])

    # Gabungkan berdasarkan ticker dan tanggal
    df_merged = df_log.merge(df_data, on=["ticker", "tanggal"], how="inner")

    # Evaluasi apakah prediksi benar: future_high <= actual_high dan future_low >= actual_low
    df_merged["benar"] = ((df_merged["actual_high"] >= df_merged["pred_high"]) &
                          (df_merged["actual_low"] <= df_merged["pred_low"]))

    # Hitung akurasi per ticker
    return df_merged.groupby("ticker")["benar"].mean().to_dict()
    
def reset_models():
    # Pola file model
    patterns = [
        "model_high_*.pkl",
        "model_low_*.pkl",
        "model_lstm_*.keras"
    ]

    total_deleted = 0
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                print(f"Dihapus: {filepath}")
                total_deleted += 1
            except Exception as e:
                print(f"Gagal menghapus {filepath}: {e}")
    
    if total_deleted == 0:
        print("Tidak ada model yang ditemukan untuk dihapus.")
    else:
        print(f"Total {total_deleted} model dihapus.")
        
# === Daftar Kutipan Motivasi ===
MOTIVATION_QUOTES = [
    "Setiap peluang adalah langkah kecil menuju kebebasan finansial.",
    "Cuan bukan tentang keberuntungan, tapi tentang konsistensi dan strategi.",
    "Disiplin hari ini, hasil luar biasa nanti.",
    "Trader sukses bukan yang selalu benar, tapi yang selalu siap.",
    "Naik turun harga itu biasa, yang penting arah portofolio naik.",
    "Fokus pada proses, profit akan menyusul.",
    "Jangan hanya lihat harga, lihat potensi di baliknya.",
    "Ketika orang ragu, itulah peluang sesungguhnya muncul.",
    "Investasi terbaik adalah pada pengetahuan dan ketenangan diri.",
    "Satu langkah hari ini lebih baik dari seribu penyesalan besok."
    "Moal rugi jalma nu talek jeung tekadna kuat.",
    "Rejeki mah moal ka tukang, asal usaha jeung sabar.",
    "Lamun hayang hasil nu beda, ulah make cara nu sarua terus.",
    "Ulah sieun gagal, sieun lamun teu nyobaan.",
    "Cuan nu leres asal ti élmu jeung kasabaran.",
    "Sabada hujan pasti aya panonpoé, sabada rugi bisa aya untung.",
    "Ngabagéakeun resiko teh bagian tina kamajuan.",
    "Jalma nu kuat téh lain nu teu pernah rugi, tapi nu sanggup bangkit deui.",
    "Ngora kudu wani nyoba, heubeul kudu wani investasi.",
    "Reureujeungan ayeuna, kabagjaan engké."
    "Niat alus, usaha terus, hasil bakal nuturkeun.",
    "Ulah ngadagoan waktu nu pas, tapi cobian ayeuna.",
    "Hirup teh kawas saham, kadang naek kadang turun, tapi ulah leungit arah.",
    "Sakumaha gede ruginya, élmu nu diala leuwih mahal hargana.",
    "Ulah beuki loba mikir, beuki saeutik tindakan.",
    "Kabagjaan datang ti tangtungan jeung harepan nu dilaksanakeun.",
    "Panghasilan teu datang ti ngalamun, tapi ti aksi jeung analisa.",
    "Sasat nu bener, bakal mawa kana untung nu lila.",
    "Tong ukur ningali batur nu untung, tapi diajar kumaha cara maranéhna usaha.",
    "Jalma sukses mah sok narima gagal minangka bagian ti perjalanan."
    "Saham bisa turun, tapi semangat kudu tetap ngora. Jalan terus, rejeki moal salah alamat.",
    "Kadang market galak, tapi inget, nu sabar jeung konsisten nu bakal panén hasilna.",
    "Cuan moal datang ti harepan hungkul, kudu dibarengan ku strategi jeung tekad.",
    "Teu aya jalan pintas ka sukses, ngan aya jalan nu jelas jeung disiplin nu kuat.",
    "Di balik koreksi aya akumulasi, di balik gagal aya élmu anyar. Ulah pundung!",
    "Sakumaha seredna pasar, nu kuat haténa bakal salamet.",
    "Rejeki teu datang ti candaan, tapi ti candak kaputusan jeung tindakan.",
    "Sugan ayeuna can untung, tapi tong hilap, tiap analisa téh tabungan pangalaman.",
    "Tenang lain berarti nyerah, tapi ngatur posisi jeung nunggu waktu nu pas.",
    "Sagalana dimimitian ku niat, dilaksanakeun ku disiplin, jeung dipanen ku waktu."
    "“Suatu saat akan datang hari di mana semua akan menjadi kenangan.” – Erza Scarlet (Fairy Tail)",
    "“Lebih baik menerima kejujuran yang pahit, daripada kebohongan yang manis.” – Soichiro Yagami (Death Note)",
    "“Jangan menyerah. Hal memalukan bukanlah ketika kau jatuh, tetapi ketika kau tidak mau bangkit lagi.” – Midorima Shintarou (Kuroko no Basuke)",
    "“Jangan khawatirkan apa yang dipikirkan orang lain. Tegakkan kepalamu dan melangkahlah ke depan.” – Izuku Midoriya (Boku no Hero Academia)",
    "“Tuhan tak akan menempatkan kita di sini melalui derita demi derita bila Ia tak yakin kita bisa melaluinya.” – Kano Yuki (Sword Art Online)",
    "“Mula-mula, kau harus mengubah dirimu sendiri atau tidak akan ada yang berubah untukmu.” – Sakata Gintoki (Gintama)",
    "“Banyak orang gagal karena mereka tidak memahami usaha yang diperlukan untuk menjadi sukses.” – Yukino Yukinoshita (Oregairu)",
    "“Kekuatan sejati dari umat manusia adalah bahwa kita memiliki kuasa penuh untuk mengubah diri kita sendiri.” – Saitama (One Punch Man)",
    "“Hidup bukanlah permainan keberuntungan. Jika kau ingin menang, kau harus bekerja keras.” – Sora (No Game No Life)",
    "“Kita harus mensyukuri apa yang kita punya saat ini karena mungkin orang lain belum tentu mempunyainya.” – Kayaba Akihiko (Sword Art Online)",
    "“Kalau kau ingin menangis karena gagal, berlatihlah lebih keras lagi sehingga kau pantas menangis ketika kau gagal.” – Megumi Takani (Samurai X)",
    "“Ketika kau bekerja keras dan gagal, penyesalan itu akan cepat berlalu. Berbeda dengan penyesalan ketika tidak berani mencoba.” – Akihiko Usami (Junjou Romantica)",
    "“Ketakutan bukanlah kejahatan. Itu memberitahukan apa kelemahanmu. Dan begitu tahu kelemahanmu, kamu bisa menjadi lebih kuat.” – Gildarts (Fairy Tail)",
    "“Untuk mendapatkan kesuksesan, keberanianmu harus lebih besar daripada ketakutanmu.” – Han Juno (Eureka Seven)",
    "“Kegagalan seorang pria yang paling sulit yaitu ketika dia gagal untuk menghentikan air mata seorang wanita.” – Kasuka Heiwajima (Durarara!)",
    "“Air mata palsu bisa menyakiti orang lain. Tapi, senyuman palsu hanya akan menyakiti dirimu sendiri.” – C.C (Code Geass)",
    "“Kita harus menjalani hidup kita sepenuhnya. Kamu tidak pernah tahu, kita mungkin sudah mati besok.” – Kaori Miyazono (Shigatsu wa Kimi no Uso)",
    "“Bagaimana kamu bisa bergerak maju kalau kamu terus menyesali masa lalu?” – Edward Elric (Fullmetal Alchemist: Brotherhood)",
    "“Jika kau seorang pria, buatlah wanita yang kau cintai jatuh cinta denganmu apa pun yang terjadi!” – Akhio (Clannad)",
    "“Semua laki-laki mudah cemburu dan bego, tapi perempuan malah menyukainya. Orang jadi bodoh saat jatuh cinta.” – Horo (Spice and Wolf)",
    "“Wanita itu sangat indah, satu senyuman mereka saja sudah menjadi sebuah keajaiban.” – Onigiri (Air Gear)",
    "“Saat kamu harus memilih satu cinta aja, pasti ada orang lain yang menangis.” – Tsubame (Ai Kora)",
    "“Aku tidak suka hubungan yang tidak jelas.” – Senjougahara (Bakemonogatari)",
    "“Cewek itu seharusnya lembut dan baik, dan bisa menyembuhkan luka di hati.” – Yoshii (Baka to Test)",
    "“Keluargamu adalah pahlawanmu.” – Sinchan (C. Sinchan)"
    "Hidup itu sederhana, kita yang membuatnya sulit. – Confucius.",
    "Hal yang paling penting adalah menikmati hidupmu, menjadi bahagia, apa pun yang terjadi. - Audrey Hepburn.",
    "Hidup itu bukan soal menemukan diri Anda sendiri, hidup itu membuat diri Anda sendiri. - George Bernard Shaw.",
    "Hidup adalah mimpi bagi mereka yang bijaksana, permainan bagi mereka yang bodoh, komedi bagi mereka yang kaya, dan tragedi bagi mereka yang miskin. - Sholom Aleichem.",
    "Kenyataannya, Anda tidak tahu apa yang akan terjadi besok. Hidup adalah pengendaraan yang gila dan tidak ada yang menjaminnya. – Eminem.",
    "Tujuan hidup kita adalah menjadi bahagia. - Dalai Lama.",
    "Hidup yang baik adalah hidup yang diinspirasi oleh cinta dan dipandu oleh ilmu pengetahuan. - Bertrand Russell.",
    "Seribu orang tua bisa bermimpi, satu orang pemuda bisa mengubah dunia. – Soekarno.",
    "Pendidikan adalah senjata paling ampuh untuk mengubah dunia. - Nelson Mandela.",
    "Usaha dan keberanian tidak cukup tanpa tujuan dan arah perencanaan. - John F. Kennedy.",
    "Dunia ini cukup untuk memenuhi kebutuhan manusia, bukan untuk memenuhi keserakahan manusia. - Mahatma Gandhi.",
    "Jika kamu berpikir terlalu kecil untuk membuat sebuah perubahan, cobalah tidur di ruangan dengan seekor nyamuk. - Dalai Lama.",
    "Anda mungkin bisa menunda, tapi waktu tidak akan menunggu. - Benjamin Franklin.",
    "Kamu tidak perlu menjadi luar biasa untuk memulai, tapi kamu harus memulai untuk menjadi luar biasa. - Zig Ziglar.",
    "Jangan habiskan waktumu memukuli dinding dan berharap bisa mengubahnya menjadi pintu. - Coco Chanel.",
    "Tidak ada yang akan berhasil kecuali kau melakukannya. - Maya Angelou.",
    "Kamu tidak bisa kembali dan mengubah awal saat kamu memulainya, tapi kamu bisa memulainya lagi dari mana kamu berada sekarang dan ubah akhirnya. - C.S Lewis.",
    "Beberapa orang memimpikan kesuksesan, sementara yang lain bangun setiap pagi untuk mewujudkannya. - Wayne Huizenga.",
    "Pekerjaan-pekerjaan kecil yang selesai dilakukan lebih baik daripada rencana-rencana besar yang hanya didiskusikan. - Peter Marshall.",
    "Kita harus berarti untuk diri kita sendiri dulu sebelum kita menjadi orang yang berharga bagi orang lain. - Ralph Waldo Emerson.",
    "Hal yang paling menyakitkan adalah kehilangan jati dirimu saat engkau terlalu mencintai seseorang. Serta lupa bahwa sebenarnya engkau juga spesial. - Ernest Hemingway.",
    "Beberapa orang akan pergi dari hidupmu, tapi itu bukan akhir dari ceritamu. Itu cuma akhir dari bagian mereka di ceritamu. - Faraaz Kazi.",
    "Cinta terjadi begitu singkat, namun melupakan memakan waktu begitu lama. - Pablo Neruda.",
    "Seseorang tak akan pernah tahu betapa dalam kadar cintanya sampai terjadi sebuah perpisahan. - Kahlil Gibran.",
    "Hubungan asmara itu seperti kaca. Terkadang lebih baik meninggalkannya dalam keadaan pecah daripada menyakiti dirimu dengan cara menyatukan mereka kembali. - D.Love.",
    "Cinta itu seperti angin. Kau tak dapat melihatnya, tapi kau dapat merasakannya. - Nicholas Sparks.",
    "Cinta adalah ketika kebahagiaan orang lain lebih penting dari kebahagiaanmu. - H. Jackson Brown.",
    "Asmara bukan hanya sekadar saling memandang satu sama lain. Tapi, juga sama-sama melihat ke satu arah yang sama. - Antoine de Saint-Exupéry.",
    "Bagaimana kau mengeja ‘cinta’? tanya Piglet. Kau tak usah mengejanya, rasakan saja, jawab Pooh. - A.A Milne.",
    "Kehidupan adalah 10 persen apa yang terjadi terhadap Anda dan 90 persen adalah bagaimana Anda meresponnya. - Lou Holtz.",
    "Satu-satunya keterbatasan dalam hidup adalah perilaku yang buruk. - Scott Hamilton.",
    "Seseorang yang berani membuang satu jam waktunya tidak mengetahui nilai dari kehidupan. - Charles Darwin.",
    "Apa yang kita pikirkan menentukan apa yang akan terjadi pada kita. Jadi jika kita ingin mengubah hidup, kita perlu sedikit mengubah pikiran kita. - Wayne Dyer.",
    "Ia yang mengerjakan lebih dari apa yang dibayar pada suatu saat nanti akan dibayar lebih dari apa yang ia kerjakan. - Napoleon Hill.",
    "Saya selalu mencoba untuk mengubah kemalangan menjadi kesempatan. - John D. Rockefeller.",
    "Seseorang yang pernah melakukan kesalahan dan tidak pernah memperbaikinya berarti ia telah melakukan satu kesalahan lagi. - Konfusius.",
    "Anda tidak akan pernah belajar sabar dan berani jika di dunia ini hanya ada kebahagiaan. - Helen Keller.",
    "Tidak apa-apa untuk merayakan kesuksesan, tapi lebih penting untuk memperhatikan pelajaran tentang kegagalan. – Bill Gates."
]

def get_random_motivation() -> str:
    return random.choice(MOTIVATION_QUOTES)

# === Eksekusi & Kirim Sinyal ===
if __name__ == "__main__":
    reset_models()
    logging.info("🚀 Memulai analisis saham...")
    max_workers = min(8, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r]

    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Backup CSV disimpan")

    top_5 = sorted(results, key=lambda x: x["take_profit"], reverse=True)[:5]
    if top_5:
        motivation = get_random_motivation()
        message = (
            f"<b>💩Hai KONTIL Clan Member💩</b>\n"
            f"<b>Apapun Yang Sedang Kalian Hadapi Saat Ini, Ingatlah...</b>\n"
            f"<b><i>{motivation}</i></b>\n\n"
            f"<b>Berikut Top 5 saham pilihan berdasarkan analisa KONTIL AI:</b>\n"
        )
        for r in top_5:
            message += (
                f"\n🔹 {r['ticker']}\n"
                f"   💰 Harga: {r['harga']:.2f}\n"
                f"   🎯 TP: {r['take_profit']:.2f}\n"
                f"   🛑 SL: {r['stop_loss']:.2f}\n"
                f"   📈 Potensi Profit: {r['profit_potential_pct']:.2f}%\n"
                f"   ✅ Probabilitas: {r['prob_success']*100:.1f}%\n"
                f"   📌 Aksi: <b>{r['aksi'].upper()}</b>\n"
            )
        send_telegram_message(message)

    logging.info("✅ Selesai.")
