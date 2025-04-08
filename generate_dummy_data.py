import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 100

dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='H')
df = pd.DataFrame({
    'Datetime': dates,
    'Close': np.random.uniform(1000, 1500, size=n),
    'SMA_5': np.random.uniform(1000, 1500, size=n),
    'SMA_10': np.random.uniform(1000, 1500, size=n),
    'EMA_5': np.random.uniform(1000, 1500, size=n),
    'EMA_10': np.random.uniform(1000, 1500, size=n),
    'RSI_14': np.random.uniform(30, 70, size=n),
    'MACD': np.random.randn(n),
    'MACD_signal': np.random.randn(n),
    'BB_Middle': np.random.uniform(1000, 1500, size=n),
    'BB_Upper': np.random.uniform(1500, 1600, size=n),
    'BB_Lower': np.random.uniform(900, 1000, size=n),
    'Stochastic_K': np.random.uniform(0, 100, size=n),
    'Stochastic_D': np.random.uniform(0, 100, size=n),
    'ATR_14': np.random.uniform(10, 20, size=n),
    'Volume': np.random.randint(1000, 10000, size=n),
    'Volume_MA20': np.random.randint(1000, 10000, size=n),
    'Tenkan_sen': np.random.uniform(1000, 1500, size=n),
    'Kijun_sen': np.random.uniform(1000, 1500, size=n),
    'Senkou_Span_A': np.random.uniform(1000, 1500, size=n),
    'Senkou_Span_B': np.random.uniform(1000, 1500, size=n),
    'Chikou_Span': np.random.uniform(1000, 1500, size=n),
    'SAR': np.random.uniform(1000, 1500, size=n),
    'Fibo_0.236': np.random.uniform(1000, 1500, size=n),
    'Fibo_0.382': np.random.uniform(1000, 1500, size=n),
    'Fibo_0.5': np.random.uniform(1000, 1500, size=n),
    'Fibo_0.618': np.random.uniform(1000, 1500, size=n),
    'Fibo_0.786': np.random.uniform(1000, 1500, size=n),
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/historical_data.csv', index=False)
print("✅ Dummy data berhasil dibuat: data/historical_data.csv")