import requests
import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_signal_to_telegram(symbol, signal):
    if not signal.get("send_signal"):
        print(f"⏸️ Tidak ada sinyal valid untuk {symbol}")
        return

    message = f"""
📈 *Sinyal Trading - {symbol}*

*Rekomendasi:* {signal['recommendation']}
*Probabilitas Naik:* {signal['predict_up_prob']:.2%}

*Harga Sekarang:* Rp{signal['last_close']:,}
*Take Profit:* Rp{signal['take_profit']:,}
*Stop Loss:* Rp{signal['stop_loss']:,}
*Risk/Reward Ratio:* {signal['risk_reward']}

⚡ _Sinyal ini otomatis dihasilkan oleh AI trading bot._

#DayTrading #SinyalSahamAI
""".strip()

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    response = requests.post(url, data=data)
    if response.status_code == 200:
        print(f"✅ Sinyal {symbol} terkirim!")
    else:
        print(f"❌ Gagal kirim sinyal {symbol}: {response.text}")
