# === trading_bot.py ===
import os, time, joblib, requests, logging
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf
from datetime import datetime, timedelta
from ta import momentum, trend, volatility, volume
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

# === Konfigurasi ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RETRAIN_INTERVAL = 7
MODEL_HIGH_PATH = "model_high.pkl"
MODEL_LOW_PATH = "model_low.pkl"
MODEL_LSTM_PATH = "model_lstm.keras"
BACKUP_CSV_PATH = "stock_data_backup.csv"
MAX_WORKERS = 7
MIN_PROBABILITY = 0.5
MIN_PROFIT_PERCENT = 1.0

# === Logging ===
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = RotatingFileHandler("trading.log", maxBytes=2_000_000, backupCount=3)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# === Daftar Saham ===
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

def send_telegram_message(message, max_retries=3):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    for _ in range(max_retries):
        try:
            response = requests.post(url, data=data, timeout=10)
            if response.ok:
                return True
        except Exception as e:
            logger.warning(f"Gagal kirim Telegram: {e}")
        time.sleep(2)
    return False

def get_stock_data(ticker, max_retries=3):
    for _ in range(max_retries):
        try:
            data = yf.download(ticker, period="3y", interval="1d", progress=False, auto_adjust=True)
            if data is not None and not data.empty and len(data) > 100:
                data["ticker"] = ticker
                return data
        except Exception as e:
            logger.error(f"Gagal ambil data {ticker}: {e}")
        time.sleep(2)
    return None

def calculate_indicators(df):
    try:
        # Volatility Indicators
        df["ATR"] = volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
        bb = volatility.BollingerBands(close=df["Close"], window=20)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()

        # Momentum Indicators
        df["RSI"] = momentum.RSIIndicator(close=df["Close"], window=14).rsi()
        macd = trend.MACD(close=df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Hist"] = macd.macd_diff()
        df["ADX"] = trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()

        # Trend Indicators
        df["SMA_50"] = trend.SMAIndicator(close=df["Close"], window=50).sma_indicator()
        df["SMA_200"] = trend.SMAIndicator(close=df["Close"], window=200).sma_indicator()

        # Volume Indicator
        df["VWAP"] = volume.VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]).volume_weighted_average_price()

        # Support and Resistance
        df["Support"] = df["Low"].rolling(window=20).min()
        df["Resistance"] = df["High"].rolling(window=20).max()

        # Future Target (for model prediction)
        df["future_high"] = df["High"].shift(-1)
        df["future_low"] = df["Low"].shift(-1)

        return df.dropna()
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03)
    model.fit(X, y)
    return model

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
    model.fit(X, y, epochs=55, batch_size=32, verbose=0)
    return model

def is_retraining_needed(path, interval_days=RETRAIN_INTERVAL):
    if not os.path.exists(path):
        return True
    last_trained = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - last_trained > timedelta(days=interval_days)

def analyze_stock(ticker):
    df = get_stock_data(ticker)
    if df is None:
        return None
    df = calculate_indicators(df)
    if df is None or df.empty:
        logger.warning(f"Data indikator kosong: {ticker}")
        return None

    features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
    df = df.dropna(subset=features + ["future_high", "future_low"])
    if df.empty:
        return None

    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]

    retrain = is_retraining_needed(MODEL_HIGH_PATH)
    if retrain:
        logger.info(f"Retrain model untuk {ticker}")
        model_high = train_lightgbm(X, y_high)
        model_low = train_lightgbm(X, y_low)
        model_lstm = train_lstm(X, y_high)
        joblib.dump(model_high, MODEL_HIGH_PATH)
        joblib.dump(model_low, MODEL_LOW_PATH)
        model_lstm.save(MODEL_LSTM_PATH)
    else:
        model_high = joblib.load(MODEL_HIGH_PATH)
        model_low = joblib.load(MODEL_LOW_PATH)

    X_last = X.iloc[[-1]]
    current_price = df["Close"].iloc[-1]
    pred_high = model_high.predict(X_last)[0]
    pred_low = model_low.predict(X_last)[0]

    profit_pct = (pred_high - current_price) / current_price * 100
    probability = np.clip((pred_high - pred_low) / (df["High"].max() - df["Low"].min()), 0, 1)

    if profit_pct < MIN_PROFIT_PERCENT or probability < MIN_PROBABILITY:
        return None

    return {
        "ticker": ticker,
        "harga": round(current_price, 2),
        "take_profit": round(pred_high, 2),
        "stop_loss": round(pred_low, 2),
        "potensi_profit": round(profit_pct, 2),
        "probabilitas": round(probability, 2),
        "aksi": "beli"
    }

def main():
    logger.info("Memulai analisis saham...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r]
    top5 = sorted(results, key=lambda x: (x["potensi_profit"], x["probabilitas"]), reverse=True)[:5]

    if top5:
        message = "<b>Top 5 Sinyal Saham Hari Ini</b>\n"
        for i, stock in enumerate(top5, 1):
            message += (
                f"\n<b>{i}. {stock['ticker']}</b>\n"
                f"Aksi: <b>{stock['aksi'].upper()}</b>\n"
                f"Harga: {stock['harga']}\n"
                f"TP: {stock['take_profit']}, SL: {stock['stop_loss']}\n"
                f"Potensi: {stock['potensi_profit']}%, Probabilitas: {stock['probabilitas']}\n"
            )
        send_telegram_message(message)
        logger.info("Sinyal dikirim ke Telegram.")
    else:
        logger.info("Tidak ada sinyal yang memenuhi kriteria hari ini.")
        send_telegram_message("Tidak ada sinyal yang memenuhi kriteria hari ini.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Terjadi kesalahan fatal: {e}")
        send_telegram_message(f"Bot trading mengalami error: {e}")
