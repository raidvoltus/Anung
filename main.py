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

# === [KONFIGURASI BOT] ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
ATR_MULTIPLIER = 2.5
RETRAIN_INTERVAL = 1
MIN_PROBABILITY = 0.8 # Filter probabilitas reward > risk
MODEL_HIGH_PATH = "model_high.pkl"
MODEL_LOW_PATH = "model_low.pkl"
MODEL_LSTM_PATH = "model_lstm.h5"
BACKUP_CSV_PATH = "stock_data_backup.csv"
LAST_TRAIN_FILE = "last_train.txt"

# === [DAFTAR SAHAM] ===
STOCK_LIST = [
    "AALI.JK", "ABBA.JK", "ABMM.JK", "ACES.JK", "ACST.JK", "ADHI.JK", "ADMF.JK", "ADMG.JK", "ADRO.JK", "AGII.JK",
    "AGRO.JK", "AKRA.JK", "AKSI.JK", "ALDO.JK", "ALKA.JK", "ALMI.JK", "AMAG.JK", "AMRT.JK", "ANDI.JK", "ANJT.JK",
    "ANTM.JK", "APIC.JK", "APLN.JK", "ARNA.JK", "ARTA.JK", "ASII.JK", "ASJT.JK", "ASRI.JK", "ASSA.JK", "ATIC.JK",
    "AUTO.JK", "BABP.JK", "BACA.JK", "BAEK.JK", "BALI.JK", "BAPA.JK", "BAPI.JK", "BATA.JK", "BBCA.JK", "BBHI.JK",
    "BBKP.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BINA.JK", "BIPP.JK", "BISI.JK", "BJBR.JK", "BJTM.JK", "BKDP.JK",
    "BKSL.JK", "BKSW.JK", "BLTA.JK", "BLTZ.JK", "BLUE.JK", "BMAS.JK", "BMRI.JK", "BMSR.JK", "BMTR.JK", "BNBA.JK",
    "BNGA.JK", "BNII.JK", "BNLI.JK", "BOBA.JK", "BOGA.JK", "BOLT.JK", "BOSS.JK", "BPFI.JK", "BPII.JK", "BPTR.JK",
    "BRAM.JK", "BRIS.JK", "BRMS.JK", "BRPT.JK", "BSDE.JK", "BSSR.JK", "BTEK.JK", "BTEL.JK", "BTON.JK", "BTPN.JK",
    "BTPS.JK", "BUDI.JK", "BUVA.JK", "BVSN.JK", "BYAN.JK", "CAKK.JK", "CAMP.JK", "CANI.JK", "CARS.JK", "CASA.JK",
    "CASH.JK", "CBMF.JK", "CEKA.JK", "CENT.JK", "CFIN.JK", "CINT.JK", "CITA.JK", "CITY.JK", "CLAY.JK", "CLEO.JK",
    "CLPI.JK", "CMNP.JK", "CMRY.JK", "CMPP.JK", "CNKO.JK", "CNTX.JK", "COCO.JK", "COWL.JK", "CPIN.JK", "CPRO.JK",
    "CSAP.JK", "CSIS.JK", "CTRA.JK", "CTTH.JK", "DEAL.JK", "DEFI.JK", "DEPO.JK", "DGIK.JK", "DIGI.JK", "DILD.JK",
    "DIVA.JK", "DKFT.JK", "DLTA.JK", "DMAS.JK", "DNAR.JK", "DOID.JK", "DSSA.JK", "DUCK.JK", "DUTI.JK", "DVLA.JK",
    "DYAN.JK", "EAST.JK", "ECII.JK", "EDGE.JK", "EKAD.JK", "ELSA.JK", "EMDE.JK", "EMTK.JK", "ENRG.JK", "ENZO.JK",
    "EPAC.JK", "ERA.JK", "ERAA.JK", "ESSA.JK", "ESTA.JK", "FAST.JK", "FASW.JK", "FILM.JK", "FISH.JK", "FITT.JK",
    "FLMC.JK", "FMII.JK", "FOOD.JK", "FORU.JK", "FPNI.JK", "GAMA.JK", "GEMS.JK", "GGRM.JK", "GJTL.JK", "GLVA.JK",
    "GOOD.JK", "GPRA.JK", "GSMF.JK", "GZCO.JK", "HDTX.JK", "HERO.JK", "HEXA.JK", "HITS.JK", "HKMU.JK", "HMSP.JK",
    "HOKI.JK", "HRUM.JK", "ICBP.JK", "IDPR.JK", "IFII.JK", "INAF.JK", "INAI.JK", "INCF.JK", "INCI.JK", "INCO.JK",
    "INDF.JK", "INDY.JK", "INKP.JK", "INPP.JK", "INTA.JK", "INTD.JK", "INTP.JK", "IPCC.JK", "IPCM.JK", "IPOL.JK",
    "IPTV.JK", "IRRA.JK", "ISAT.JK", "ITMG.JK", "JAST.JK", "JAWA.JK", "JGLE.JK", "JKON.JK", "JPFA.JK", "JSMR.JK",
    "KAEF.JK", "KARW.JK", "KBLI.JK", "KBLM.JK", "KDSI.JK", "KIAS.JK", "KIJA.JK", "KINO.JK", "KLBF.JK", "KMTR.JK",
    "LEAD.JK", "LIFE.JK", "LINK.JK", "LPKR.JK", "LPPF.JK", "LUCK.JK", "MAIN.JK", "MAPB.JK", "MAPA.JK", "MASA.JK",
    "MCAS.JK", "MDKA.JK", "MEDC.JK", "MFIN.JK", "MIDI.JK", "MIRA.JK", "MITI.JK", "MKNT.JK", "MLPL.JK", "MLPT.JK",
    "MNCN.JK", "MPPA.JK", "MPRO.JK", "MTDL.JK", "MYOR.JK", "NATO.JK", "NELY.JK", "NFCX.JK", "NISP.JK", "NRCA.JK",
    "OKAS.JK", "OMRE.JK", "PANI.JK", "PBID.JK", "PCAR.JK", "PDES.JK", "PEHA.JK", "PGAS.JK", "PJAA.JK", "PMJS.JK",
    "PNBN.JK", "PNLF.JK", "POLA.JK", "POOL.JK", "PPGL.JK", "PPRO.JK", "PSSI.JK", "PTBA.JK", "PTIS.JK", "PWON.JK",
    "RAJA.JK", "RDTX.JK", "REAL.JK", "RICY.JK", "RIGS.JK", "ROTI.JK", "SAME.JK", "SAPX.JK", "SCCO.JK", "SCMA.JK",
    "SIDO.JK", "SILO.JK", "SIMP.JK", "SIPD.JK", "SMBR.JK", "SMCB.JK", "SMDR.JK", "SMGR.JK", "SMKL.JK", "SMRA.JK",
    "SMSM.JK", "SOCI.JK", "SQMI.JK", "SRAJ.JK", "SRTG.JK", "STAA.JK", "STTP.JK", "TALF.JK", "TARA.JK", "TBIG.JK",
    "TCID.JK", "TIFA.JK", "TINS.JK", "TKIM.JK", "TLKM.JK", "TOTO.JK", "TPIA.JK", "TRIM.JK", "TURI.JK", "ULTJ.JK",
    "UNIC.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSBP.JK", "WSKT.JK", "YPAS.JK", "ZINC.JK"
]

# === [LOGGING] ===
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler = RotatingFileHandler("trading.log", maxBytes=33*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

# === [FUNGSI TELEGRAM] ===
def send_telegram_message(message):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"})
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# === [RETRAIN CONTROL] ===
def should_retrain():
    if not os.path.exists(LAST_TRAIN_FILE): return True
    try:
        with open(LAST_TRAIN_FILE, "r") as f:
            last_time = datetime.strptime(f.read().strip(), "%Y-%m-%d")
        return (datetime.now() - last_time).days >= RETRAIN_INTERVAL
    except Exception as e:
        logging.error(f"Gagal membaca file retrain: {e}")
        return True

def update_last_train_time():
    try:
        with open(LAST_TRAIN_FILE, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d"))
    except Exception as e:
        logging.error(f"Gagal menyimpan waktu retrain: {e}")

# === [DATA & INDIKATOR] ===
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="60d", interval="30m")
        if data is not None and not data.empty and len(data) >= 200:
            data["ticker"] = ticker
            return data
        logging.warning(f"Data kosong/kurang: {ticker}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

def calculate_indicators(df):
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=10).average_true_range()
    df["MACD"] = trend.MACD(df["Close"]).macd()
    df["MACD_Hist"] = trend.MACD(df["Close"]).macd_diff()
    bb = volatility.BollingerBands(df["Close"], window=12)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["Support"] = df["Low"].rolling(window=24).min()
    df["Resistance"] = df["High"].rolling(window=24).max()
    df["RSI"] = momentum.RSIIndicator(df["Close"], window=10).rsi()
    df["SMA_50"] = trend.SMAIndicator(df["Close"], window=24).sma_indicator()
    df["SMA_200"] = trend.SMAIndicator(df["Close"], window=48).sma_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=10).adx()
    df["future_high"] = df["High"].shift(-1)
    df["future_low"] = df["Low"].shift(-1)
    return df.dropna()

# === [TRAINING MODEL] ===
def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    return model

def train_lstm(X, y):
    X_reshaped = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_reshaped.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_reshaped, y, epochs=55, batch_size=32, verbose=0)
    return model

# === [ANALISIS SAHAM] ===
def analyze_stock(ticker):
    df = get_stock_data(ticker)
    if df is None: return None
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

    X_input = X.iloc[-1:]
    pred_high_lgb = model_high.predict(X_input.values)[0]
    pred_high_lstm = model_lstm.predict(np.reshape(X_input.values, (1, X_input.shape[1], 1)))[0][0]
    pred_high = (pred_high_lgb + pred_high_lstm) / 2
    pred_low = model_low.predict(X_input.values)[0]

    current_price = df["Close"].iloc[-1]
    risk = current_price - pred_low
    reward = pred_high - current_price
    prob = reward / (reward + risk) if reward + risk > 0 else 0

    if risk <= 0 or prob < MIN_PROBABILITY: return None

    return {
        "ticker": ticker,
        "harga": round(current_price, 2),
        "take_profit": round(pred_high, 2),
        "stop_loss": round(pred_low, 2),
        "aksi": "beli",
        "potensi": round(reward, 2),
        "probabilitas": round(prob * 100, 2)
    }

# === [MAIN EKSEKUSI] ===
if __name__ == "__main__":
    logging.info("🚀 Memulai analisis saham...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))
    results = [r for r in results if r and r["aksi"] == "beli"]

    top_5 = sorted(results, key=lambda x: x["potensi"], reverse=True)[:5]
    if top_5:
        message = "<b>📊 Hai Para Kontil, Berikut Top 5 Sinyal Trading Hari Ini:</b>\n"
        for r in top_5:
            message += (f"\n🔹 <b>{r['ticker']}</b>\n"
                        f"   Harga: {r['harga']}\n"
                        f"   TP: {r['take_profit']} | SL: {r['stop_loss']}\n"
                        f"   Potensi: {r['potensi']} | Prob: {r['probabilitas']}%\n"
                        f"   Aksi: <b>{r['aksi'].upper()}</b>\n")
        send_telegram_message(message)
    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Selesai dan data disimpan.")
