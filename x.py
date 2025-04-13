import requests

TOKEN = "PASTE_TOKEN_BOT_KAMU_DI_SINI"
URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

try:
    response = requests.get(URL)
    response.raise_for_status()
    data = response.json()

    if data.get("ok"):
        results = data.get("result", [])
        for update in results:
            message = update.get("message")
            if message:
                chat = message.get("chat", {})
                chat_id = chat.get("id")
                chat_title = chat.get("title", "Personal Chat")
                print(f"Chat ID: {chat_id} | Nama Chat: {chat_title}")
    else:
        print("Gagal mengambil data. Periksa token atau status bot.")
except requests.exceptions.RequestException as e:
    print(f"Terjadi kesalahan: {e}")