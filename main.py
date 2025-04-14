import os, time, joblib, requests, logging
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf
import matplotlib.pyplot as plt
from ta import momentum, trend, volatility, volume
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

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
ATR_MULTIPLIER = 2.5
RETRAIN_INTERVAL = 7

# --- [STOCK LIST] ---
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

# --- [UTILITY FUNCTIONS] ---
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.error("Telegram token atau chat_id tidak ditemukan. Pastikan environment variable TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID sudah di-set.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, data=data)
        if response.status_code == 33:
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
        else:
            logging.warning(f"Data kosong atau kurang: {ticker}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

def calculate_indicators(df):
    # --- Volatilitas dan Trend ---
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=5).average_true_range()
    macd = trend.MACD(df["Close"], window_slow=12, window_fast=6, window_sign=3)
    df["MACD"] = macd.macd()
    df["Signal_Line"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    bb = volatility.BollingerBands(df["Close"], window=10)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()

    # --- Support & Resistance ---
    df["Support"] = df["Low"].rolling(window=10).min()
    df["Resistance"] = df["High"].rolling(window=10).max()

    # --- Momentum ---
    stoch = momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=7)
    df["%K"] = stoch.stoch()
    df["%D"] = stoch.stoch_signal()
    df["RSI"] = momentum.RSIIndicator(df["Close"], window=7).rsi()
    df["CCI"] = momentum.CCIIndicator(df["High"], df["Low"], df["Close"], window=10).cci()
    df["MOM"] = momentum.MomentumIndicator(df["Close"], window=5).momentum()

    # --- Moving Average & VWAP ---
    df["SMA_50"] = trend.SMAIndicator(df["Close"], window=5).sma_indicator()
    df["SMA_200"] = trend.SMAIndicator(df["Close"], window=10).sma_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()

    # --- ADX ---
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=5).adx()

    # --- Waktu ---
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    # --- Target Prediction ---
    df["future_high"] = df["High"].shift(-1)
    df["future_low"] = df["Low"].shift(-1)
    df["target_up"] = ((df["future_high"] - df["Close"]) / df["Close"] > 0.005).astype(int)

    return df.dropna()

# --- [MODEL TRAINING] ---
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
    
    # Reshape ke format (samples, timesteps, features)
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Build model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output: prediksi future_high atau future_low

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train
    model.fit(X_lstm, y, 
              epochs=50, 
              batch_size=32, 
              validation_split=0.2, 
              callbacks=[early_stop],
              verbose=1)

    return model

# --- [MAIN ANALYSIS FUNCTION] ---
def analyze_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df is None or df.empty:
            logging.warning(f"Data kosong untuk {ticker}, lewati analisis.")
            return None

        df = calculate_indicators(df)
        features = [
            "Close", "ATR", "RSI", "MACD", "MACD_Hist", "%K", "%D", "CCI", "MOM",
            "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "Support", "Resistance",
            "VWAP", "ADX", "hour", "day_of_week"
        ]
        df = df.dropna(subset=features + ["future_high", "future_low"])
        X = df[features]
        y_high = df["future_high"]
        y_low = df["future_low"]
        y_binary = (df["future_high"] > df["Close"]).astype(int)

        X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2)
        _, _, y_train_low, _ = train_test_split(X, y_low, test_size=0.2)
        X_train_cls, _, y_train_cls, _ = train_test_split(X, y_binary, test_size=0.2)

        retrain = True
        if os.path.exists(model_high_path) and os.path.exists(model_low_path) and \
           os.path.exists(model_cls_path) and os.path.exists(model_lstm_path):
            last_modified = datetime.fromtimestamp(os.path.getmtime(model_high_path))
            if (datetime.now() - last_modified).days < RETRAIN_INTERVAL:
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

        pred_high = model_high.predict(X.iloc[-1:])[0]
        pred_low = model_low.predict(X.iloc[-1:])[0]
        prob_up = model_cls.predict_proba(X.iloc[-1:])[0][1]
        current_price = df["Close"].iloc[-1]

        # Validasi probabilitas
        if prob_up < 0.8:
            logging.warning(f"Probabilitas naik terlalu rendah untuk {ticker} (prob_up: {prob_up})")
            return None

        # Menambahkan margin minimal (misal 1%) agar perbandingan TP/SL lebih realistis
        margin = 0.01  # 1% margin

        # Validasi kondisi market: hindari sideways
        boll_width = (df["BB_Upper"].iloc[-1] - df["BB_Lower"].iloc[-1]) / df["Close"].iloc[-1]
        if df["ADX"].iloc[-1] < 15 or boll_width < 0.01:
            logging.warning(f"Market sideways atau volatilitas rendah untuk {ticker}")
            return None
        if pred_high <= current_price * (1 + margin) or pred_low >= current_price * (1 - margin):
            logging.warning(
                f"Sinyal tidak valid untuk {ticker} - TP: {pred_high}, SL: {pred_low}, Harga: {current_price}"
            )
            return None

        take_profit = pred_high
        stop_loss = pred_low
        profit_pct = round((take_profit - current_price) / current_price * 100, 2)

        return {
            "ticker": ticker,
            "harga": current_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "aksi": "buy",
            "profit_pct": profit_pct,
            "probability": prob_up
        }
    except Exception as e:
        logging.error(f"Error menganalisis saham {ticker}: {e}")
        return None

def plot_probability_distribution(all_results):
    probs = [r["probability"] for r in all_results if r]
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(0.75, color='red', linestyle='--', label='Threshold 0.75')
    plt.title("Distribusi Probabilitas Kenaikan Saham")
    plt.xlabel("Probabilitas Naik")
    plt.ylabel("Jumlah Saham")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("probability_distribution distribution.png")
    plt.close()

# --- [MAIN EXECUTION] ---
if __name__ == "__main__":
    logging.info("🚀 Memulai analisis saham...")
    with ThreadPoolExecutor(max_workers=7) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))
    results = [r for r in results if r]

    if results:
        plot_probability_distribution(results)
        top_5 = sorted(results, key=lambda x: x["take_profit"], reverse=True)[:5]
        message = "<b>🍆Kontil news: Dukun pasar saham kita kesurupan lagi! Bisikan gaib baru mendarat… katanya harga bakal ngaceng. Tapi ingat, mbah cuma bantu, kamu yang ambil risiko, kontil💩📊 :</b>\n"
        for r in top_5:
            message += (
                f"\n🔹 {r['ticker']}\n   💰 Harga: {r['harga']:.2f}\n   "
                f"🎯 TP: {r['take_profit']:.2f}\n   🛑 SL: {r['stop_loss']:.2f}\n   "
                f"📌 Aksi: <b>{r['aksi'].upper()}</b>\n   📈 Potensi Profit: {r['profit_pct']}%\n"
                f"   ✅ Probabilitas: {r['probability']*100:.2f}%\n"
            )
        send_telegram_message(message)
        logging.info("✅ Sinyal berhasil dikirim ke Telegram.")
    else:
        logging.info("⚠️ Tidak ada sinyal yang memenuhi syarat hari ini.")

    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info(f"✅ Data disimpan ke {BACKUP_CSV_PATH}")
