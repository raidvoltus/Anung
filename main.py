import pandas as pd
from data_fetcher import fetch_all_data
from model_predictor import predict_signal

saham_list = ["BBRI.JK", "BBCA.JK", "TLKM.JK", "ASII.JK", "BMRI.JK"]

print("\n[INFO] Mulai analisis saham...\n")

for kode in saham_list:
    print(f"[INFO] Memproses: {kode}")

    try:
        df = fetch_all_data(kode)

        if df is None or df.empty or df.shape[0] < 100:
            print(f"[WARNING] Data tidak cukup untuk {kode}")
            continue

        signal = predict_signal(df)

        if signal["send_signal"]:
            print("\n=== SINYAL BELI TERVALIDASI ===")
            print(f"Saham        : {kode}")
            print(f"Harga Tutup  : {signal['last_close']}")
            print(f"Take Profit  : {signal['take_profit']}")
            print(f"Stop Loss    : {signal['stop_loss']}")
            print(f"Probabilitas : {round(signal['predict_up_prob']*100, 2)}%")
            print(f"Risk/Reward  : {signal['risk_reward']}")
            print(f"Rekomendasi  : {signal['recommendation']}")
            print("===============================\n")
        else:
            print(f"[INFO] Tidak ada sinyal valid untuk {kode} (Prob: {round(signal['predict_up_prob']*100,2)}%, RR: {round(signal['risk_reward'],2)})\n")

    except Exception as e:
        print(f"[ERROR] Gagal memproses {kode}: {e}\n")

print("[INFO] Analisis selesai.")
