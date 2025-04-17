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

## Kebutuhan Lingkungan
- Python 3.8+
- Library: `yfinance`, `ta`, `lightgbm`, `tensorflow`, `pandas`, `numpy`, `joblib`, `requests`

## Instalasi
```bash
pip install -r requirements.txt
