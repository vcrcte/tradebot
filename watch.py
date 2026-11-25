"""Surveillance d'actions pour un achat unique à long terme : prévient quand c'est un bon moment pour acheter.

Ne passe aucun ordre. Lancé par GitHub Actions (.github/workflows/watch.yml).
Entrées : watchlist.yaml. Sorties : alerts_watch.txt (alertes à publier), watch_state.json (anti-doublon),
docs/watchlist.md (état de chaque action à la dernière vérification).

Usage : python watch.py            (passage normal)
        python watch.py --essai    (affiche l'état et le nombre de creux signalés sur 2 ans, sans rien enregistrer)
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(ROOT, "watch_state.json")
ALERTS = os.path.join(ROOT, "alerts_watch.txt")
STATUS = os.path.join(ROOT, "docs", "watchlist.md")


def load_prices(ticker):
    df = yf.download(ticker, period="3y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def indicators(close, r):
    j = r["jours_reference"]
    high = close.rolling(j, min_periods=20).max()
    dd = close / high - 1                                     # baisse depuis le plus haut récent (négatif)
    return pd.DataFrame({"close": close, "high": high, "dd": dd, "rsi": rsi(close),
                         "signal": dd <= -r["baisse_min"],
                         "sma200": close.rolling(200, min_periods=120).mean()})


def should_alert(prev, day, dd, r):
    """Anti-doublon : une alerte par baisse, sauf si elle s'aggrave nettement ou après une pause."""
    if not prev:
        return True
    days = (pd.Timestamp(day) - pd.Timestamp(prev["date"])).days
    return days >= r["pause_jours"] or dd <= prev["dd"] - r["aggravation"]


def history_count(ind, r):
    """Nombre d'alertes de baisse qu'aurait envoyées le bot sur les 2 dernières années (pour calibrer)."""
    prev, n = None, 0
    for day, row in ind.iloc[-504:].iterrows():
        if row.signal and should_alert(prev, day, row.dd, r):
            prev, n = {"date": str(day.date()), "dd": row.dd}, n + 1
    return n


def fmt_pct(x):
    return f"{x * 100:+.0f} %".replace("+-", "-")


def main():
    essai = "--essai" in sys.argv
    cfg = yaml.safe_load(open(os.path.join(ROOT, "watchlist.yaml"), encoding="utf-8"))
    r = cfg["regles"]
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    alerts, rows, today = [], [], datetime.now(timezone.utc).date()
    mois = round(r["jours_reference"] / 21)

    for a in cfg["actions"]:
        nom, t = a["nom"], a["ticker"]
        if a.get("achete"):
            rows.append(f"| {nom} | {t} | achetée | | |")
            print(f"{nom:20s} {t:9s} déjà achetée : plus surveillée")
            continue
        st = state.setdefault(t, {})
        st.setdefault("debut", str(today))
        try:
            close = load_prices(t)
        except Exception:  # une action en panne ne doit pas bloquer les autres
            close = None
        if close is None or len(close) < 30:
            msg = f"⚠️ {nom} ({t}) : cours indisponibles, à vérifier."
            print(msg)
            rows.append(f"| {nom} | {t} | indisponible | | |")
            if not essai and st.get("erreur") != str(today):
                alerts.append(msg)
                st["erreur"] = str(today)
            continue
        ind = indicators(close, r).dropna(subset=["dd"])
        last, day = ind.iloc[-1], ind.index[-1]
        age = (today - day.date()).days
        tendance = "" if math.isnan(last.sma200) else ("au-dessus" if last.close >= last.sma200 else "sous")
        prev = st.get("alerte")
        attente = (today - pd.Timestamp(st["debut"]).date()).days
        etat = "**ACHAT**" if last.signal else f"attend une baisse (jour {attente}/{r['delai_max_jours']})"
        if prev or st.get("feu_vert"):
            etat += f", déjà signalée le {(prev or {}).get('date') or st.get('feu_vert')}"
        rows.append(f"| {nom} | {t} | {last.close:.2f} | {fmt_pct(last.dd)} | {etat} |")
        print(f"{nom:20s} {t:9s} cours {last.close:9.2f}  vs plus haut {mois} mois {fmt_pct(last.dd):>6s}  "
              f"RSI {last.rsi:3.0f}  moy. 200 j : {tendance or '?'}  dernier cours {day.date()}  état : {etat}"
              + (f"  | alertes de baisse sur 2 ans : {history_count(ind, r)}" if essai else ""))
        if age > 5:
            if not essai:
                alerts.append(f"⚠️ {nom} ({t}) : dernier cours du {day.date()}, données figées ?")
            continue
        if last.signal and should_alert(prev, day, last.dd, r):
            msg = (f"📉 **{nom}** ({t}) : bon moment pour acheter. Cours {last.close:.2f}, {fmt_pct(last.dd)} sous son "
                   f"plus haut des {mois} derniers mois ({last.high:.2f}).")
            if tendance == "sous":
                msg += (" Elle est aussi sous sa moyenne 200 jours (tendance baissière) : vérifie qu'il n'y a pas "
                        "de mauvaise nouvelle derrière.")
            if prev:
                msg += f" (Déjà signalée le {prev['date']} à {fmt_pct(prev['dd'])} : elle a encore baissé depuis.)"
            alerts.append(msg)
            st["alerte"] = {"date": str(day.date()), "dd": round(float(last.dd), 4)}
        elif not prev and not st.get("feu_vert") and attente >= r["delai_max_jours"]:
            alerts.append(f"🟢 **{nom}** ({t}) : pas de baisse notable depuis {attente} jours (cours "
                          f"{last.close:.2f}, à {fmt_pct(last.dd)} de son plus haut des {mois} derniers mois). "
                          "Sur plusieurs années, attendre davantage n'apporte pas grand-chose : tu peux acheter.")
            st["feu_vert"] = str(today)

    if essai:
        print("\n".join(alerts) or "Aucune alerte.")
        return
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(f"# Actions surveillées\n\nDernière vérification : {datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC\n\n"
                f"| Action | Ticker | Cours | vs plus haut {mois} mois | État |\n|---|---|---|---|---|\n"
                + "\n".join(rows) + f"\n\nAchat unique, long terme : une alerte part dès qu'une action est au moins "
                f"{r['baisse_min']:.0%} sous son plus haut des {mois} derniers mois. Sans baisse au bout de "
                f"{r['delai_max_jours']} jours, une alerte « tu peux acheter » part quand même.\n")
    json.dump(state, open(STATE, "w"), indent=1, ensure_ascii=False)
    if alerts:
        open(ALERTS, "w", encoding="utf-8").write("\n".join(alerts) + "\n")
    print("\n".join(alerts) or "Aucune alerte.")


if __name__ == "__main__":
    main()
