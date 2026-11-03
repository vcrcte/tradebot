import os

import requests


class Telegram:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat = os.getenv("TELEGRAM_CHAT_ID")

    def __call__(self, msg: str):
        print(msg)
        alerts = os.getenv("ALERTS_FILE")  # utilisé sur GitHub : alertes publiées en commentaire d'issue
        if alerts:
            with open(alerts, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        if not (self.token and self.chat):
            return
        try:
            requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          json={"chat_id": self.chat, "text": msg}, timeout=10)
        except requests.RequestException as e:
            print(f"[telegram] échec d'envoi : {e}")
