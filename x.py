import requests

TOKEN = "7627594999:AAE-6GMStrcSgfBylJoE1PoKStOgLDDlcxc"  # Ganti dengan token bot kamu
URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

response = requests.get(URL)
data = response.json()

# Menampilkan data update
print(data)

# Mencari chat ID grup
if "result" in data:
    for update in data["result"]:
        if "message" in update:
            chat_id = update["message"]["chat"]["id"]
            print(f"Chat ID grup: {chat_id}")
