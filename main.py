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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        data = stock.history(period="60d", interval="30m")
        if data is not None and not data.empty and len(data) >= 200:
            data["ticker"] = ticker
            return data
        else:
            logging.warning(f"Data kosong atau kurang: {ticker}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

def calculate_indicators(df):
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=10).average_true_range()
    macd = trend.MACD(df["Close"])
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

# Load model dan scaler
model_high = joblib.load('model_high.pkl')
model_low = joblib.load('model_low.pkl')

def generate_signal(df):
    signals = []
    
    # Ambil data terbaru
    atr = df["ATR"].iloc[-1]
    last_close = df["Close"].iloc[-1]
    support = df["Support"].iloc[-1]
    resistance = df["Resistance"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    
    if rsi > 70:  # Kondisi Overbought → Sinyal SELL (short)
        # Hitung TP dan SL berdasarkan ATR untuk posisi SELL
        tp_sell_atr = last_close - 2 * atr  # Target hasil jika harga turun
        sl_sell_atr = last_close + atr      # Batas rugi jika harga naik
        
        # Untuk posisi SELL, TP idealnya tidak lebih rendah dari support
        # dan SL hendaknya tidak lebih tinggi dari resistance.
        # Gunakan:
        #   TP = nilai yang lebih dekat dengan harga (lebih tinggi) antara support dan perhitungan ATR
        #   SL = nilai yang lebih dekat dengan harga (lebih rendah) antara resistance dan perhitungan ATR
        sell_tp = max(support, tp_sell_atr)
        sell_sl = min(resistance, sl_sell_atr)
        
        signals.append({
            "signal": "SELL",
            "tp": sell_tp,
            "sl": sell_sl
        })
        
    elif rsi < 30:  # Kondisi Oversold → Sinyal BUY (long)
        # Hitung TP dan SL berdasarkan ATR untuk posisi BUY
        tp_buy_atr = last_close + 2 * atr  # Target hasil jika harga naik
        sl_buy_atr = last_close - atr      # Batas rugi jika harga turun
        
        # Untuk posisi BUY, TP idealnya tidak lebih tinggi dari resistance
        # dan SL hendaknya tidak lebih rendah dari support.
        # Gunakan:
        #   TP = nilai yang lebih dekat ke harga (lebih rendah) antara resistance dan perhitungan ATR
        #   SL = nilai yang lebih dekat ke harga (lebih tinggi) antara support dan perhitungan ATR
        buy_tp = min(resistance, tp_buy_atr)
        buy_sl = max(support, sl_buy_atr)
        
        signals.append({
            "signal": "BUY",
            "tp": buy_tp,
            "sl": buy_sl
        })
        
    else:  # Tidak ada sinyal yang memenuhi syarat
        signals.append({
            "signal": "HOLD",
            "tp": None,
            "sl": None
        })
    
    return signals
    
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

# --- [MAIN ANALYSIS FUNCTION] ---
def analyze_stock(ticker):
    try:
        df = get_stock_data(ticker)

        if df is None or df.empty:
            logging.warning(f"Data kosong untuk {ticker}, lewati analisis.")
            return None

        df = calculate_indicators(df)

        features = ["Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_50", "SMA_200", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX"]
        df = df.dropna(subset=features + ["future_high", "future_low"])
        X = df[features]
        if X.empty or X.iloc[-1:].empty:
            logging.warning(f"Data fitur kosong atau tidak cukup untuk {ticker}")
            return None
            
        y_high = df["future_high"]
        y_low = df["future_low"]
        y_binary = (df["future_high"] > df["Close"]).astype(int)

        X_train, _, y_train_high, _ = train_test_split(X, y_high, test_size=0.2)
        _, _, y_train_low, _ = train_test_split(X, y_low, test_size=0.2)
        X_train_cls, _, y_train_cls, _ = train_test_split(X, y_binary, test_size=0.2)

        retrain = True
        if os.path.exists(model_high_path) and os.path.exists(model_low_path) and os.path.exists(model_cls_path) and os.path.exists(model_lstm_path):
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

        if X.empty or X.iloc[-1:].empty:
            logging.warning(f"Data fitur kosong atau tidak cukup untuk prediksi {ticker}")
            return None
        pred_high = model_high.predict(X.iloc[-1:])[0]
        pred_low = model_low.predict(X.iloc[-1:])[0]
        prob_up = model_cls.predict_proba(X.iloc[-1:])[0][1]
        current_price = df["Close"].iloc[-1]

        # Simpan model dan scaler
        joblib.dump(model_high, 'model_high.pkl')
        joblib.dump(model_low, 'model_low.pkl')

        # Validasi probabilitas
        if prob_up > 0.6:
            prediksi = prediksi_sinyal(ticker, X.iloc[-1:].values, current_price)
            if prediksi:
                pesan = (f"<b>Sinyal Deteksi: {ticker}</b>\n"
                         f"Harga Saat Ini: {current_price:.2f}\n"
                         f"Target Profit (TP): {prediksi['TP']:.2f}\n"
                         f"Stop Loss (SL): {prediksi['SL']:.2f}\n"
                         f"Probabilitas Naik: {prob_up:.2%}")
                send_telegram_message(pesan)
                return prediksi
            else:
                logging.info(f"Tidak ada sinyal valid untuk {ticker}")
        else:
            logging.info(f"Probabilitas naik terlalu rendah untuk {ticker} ({prob_up:.2%})")

    except Exception as e:
        logging.error(f"Error saat analisis {ticker}: {e}")

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
        message = "<b>🍆Kontil news: Dukun pasar saham kita kesurupan lagi! Berikut bisikan gaib buat kontilers yang masih percaya hidup itu keras, tapi kita masih bisa lebih keras!💩  :</b>\n"
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
