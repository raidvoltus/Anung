import pandas as pd
import numpy as np
import yfinance as yf

# Fungsi download data historis
def download_data(symbol):
    df = yf.download(symbol, period='6mo', interval='1h', progress=False)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

# Fungsi menghitung indikator teknikal
def calculate_indicators(df):
    df = df.copy()

    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-10))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()

    # Volume Moving Average
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # Ichimoku Cloud
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)

    df['Chikou_Span'] = df['Close'].shift(-26)

    # Parabolic SAR (sederhana)
    df['SAR'] = df['Close'].shift(1)

    # Fibonacci Retracement
    max_price = df['Close'].rolling(window=50).max()
    min_price = df['Close'].rolling(window=50).min()
    diff = max_price - min_price
    df['Fibo_0.236'] = max_price - 0.236 * diff
    df['Fibo_0.382'] = max_price - 0.382 * diff
    df['Fibo_0.5'] = max_price - 0.5 * diff
    df['Fibo_0.618'] = max_price - 0.618 * diff
    df['Fibo_0.786'] = max_price - 0.786 * diff

    # Bersihkan kolom sementara
    df.drop(['H-L', 'H-PC', 'L-PC', 'BB_Std'], axis=1, inplace=True, errors='ignore')
    df.dropna(inplace=True)

    # Pastikan semua kolom scalar (bukan array/list)
    for col in df.columns:
        if isinstance(df[col].iloc[-1], (np.ndarray, list)):
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)

    return df
    # Tambahkan di bagian paling bawah file data_fetcher.py
def fetch_data_with_indicators(symbol):
    df = download_data(symbol)
    df = calculate_indicators(df)
    return df
