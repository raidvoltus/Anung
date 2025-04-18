# Stock Prediction & Signal Bot (IDX)

Bot ini secara otomatis melakukan analisis teknikal, pelatihan model prediktif, dan mengirimkan sinyal trading harian ke Telegram berdasarkan probabilitas dan estimasi potensi profit.

## Fitur Utama

### 1. **Pengambilan Data Historis**
- Menggunakan `yfinance` untuk mengambil data harga saham dari Bursa Efek Indonesia (IDX).
- Interval: `1h` (satu jam), Periode: `6 bulan` terakhir.

### 2. **Perhitungan Indikator Teknikal**
Menggunakan library `ta` untuk menghitung berbagai indikator:
- **Volatilitas**: ATR, Bollinger Bands
- **Momentum**: RSI, Stochastic
- **Trend**: MACD, SMA (20, 50, 200), ADX
- **Volume**: VWAP
- **Support/Resistance**: Rolling min/max 20 bar

### 3. **Prediksi Harga Masa Depan**
Bot melakukan prediksi `future_high` dan `future_low` untuk 6 jam ke depan:
- `future_high`: harga tertinggi dalam 6 jam mendatang
- `future_low`: harga terendah dalam 6 jam mendatang

### 4. **Model Prediktif**
- **LightGBM**: Digunakan untuk memprediksi harga tertinggi dan terendah mendatang.
- **LSTM (Keras/TensorFlow)**: Alternatif model berbasis deep learning.
- Model dilatih otomatis per ticker dan disimpan secara lokal.

### 5. **Filtering Awal Saham**
Bot akan melewati saham yang:
- Harga terakhir < 50
- Volume rata-rata 20 candle terakhir < 10.000
- Volatilitas (ATR/Close) < 0.5%

### 6. **Evaluasi Probabilitas Keberhasilan**
- Menghitung akurasi arah prediksi terhadap harga penutupan saat ini.
- Sinyal dikirim hanya jika probabilitas prediksi > 60% untuk `high` dan `low`.

### 7. **Estimasi Potensi Profit**
- Potensi profit dihitung dalam persen:
  - Aksi Beli: `(TP - Harga) / Harga * 100`
  - Aksi Jual: `(Harga - SL) / Harga * 100`
- Disertakan dalam sinyal Telegram.

### 8. **Pengiriman Sinyal Telegram**
- Sinyal dikirimkan dalam format:
  - Harga saat ini
  - Target profit (`TP`)
  - Stop loss (`SL`)
  - Probabilitas keberhasilan
  - Potensi profit (%)
  - Aksi: **BELI** atau **JUAL**
- Hanya dikirim top 5 saham dengan potensi tertinggi.

### 9. **Multi-threading dan Logging**
- Menggunakan `ThreadPoolExecutor` untuk analisis paralel multi-saham.
- Logging disimpan dalam `trading.log` dengan rotasi otomatis.

### 10. **Backup Otomatis**
- Hasil analisis disimpan dalam file `stock_data_backup.csv` setiap hari.

# Anung AI — Automated Stock Trading Bot

Anung AI adalah sistem trading saham otomatis berbasis Python yang menggunakan LightGBM & LSTM untuk memprediksi harga tertinggi dan terendah hari berikutnya. Bot ini dilengkapi dengan:

- **Data ingestion** via Yahoo Finance (`yfinance`) interval 1‑jam  
- **Lebih dari 20 indikator teknikal** (ATR, MACD, RSI, SMA/EMA, VWAP, CCI, Momentum, dll.)  
- **Filter probabilitas arah** prediksi (hanya sinyal dengan akurasi ≥ 60 %)  
- **Logging & evaluasi performa** terhadap realisasi harga  
- **Retraining otomatis** jika akurasi model per‑ticker turun di bawah 90 %  
- **Auto‑reset model** saat struktur fitur berubah  
- **Notifikasi sinyal** lewat Telegram Bot API  
- **CI/CD** dengan GitHub Actions untuk penjadwalan, eksekusi, dan push otomatis  

---

## 📋 Fitur Utama

1. **Prediksi Besok**  
   - Target: `future_high`, `future_low` besok (6 candle 1‑jam)  
   - Fitur agregat harian: `daily_avg`, `daily_std`, `daily_range`  

2. **Self‑Learning & Auto‑Retrain**  
   - Logging prediksi & realisasi (`prediksi_log.csv`)  
   - Evaluasi akurasi per-ticker (≥ 90 %)  
   - Retrain LightGBM & LSTM saat performa menurun

3. **Auto‑Reset Model**  
   - Hash struktur `features` disimpan di `features_hash.json`  
   - Model lama dihapus & retrain otomatis setiap kali `main.py` dijalankan jika fitur berubah  

4. **Telegram Notification**  
   - Ekstrak sinyal dari log  
   - Kirim pesan ke CHAT_ID via Bot TOKEN  

5. **CI/CD with GitHub Actions**  
   - Jadwal: Senin–Jumat pukul 11:00 & 15:00 WIB  
   - Workflow: run bot → publish sinyal → pull‑rebase & push repo  

---

## 🚀 Instalasi & Setup

1. **Clone repository**  
   ```bash
   git clone https://github.com/raidvoltus/Anung.git
   cd Anung
  
## Kebutuhan Lingkungan
- Python 3.8+
- Library: `yfinance`, `ta`, `lightgbm`, `tensorflow`, `pandas`, `numpy`, `joblib`, `requests`

## Instalasi
```bash
pip install -r requirements.txt
