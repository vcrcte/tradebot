"""Surveillance d'actions pour achat long terme : prévient quand une action de la liste fait un creux.

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


def indicators(close):
    high6m = close.rolling(126, min_periods=20).max()
    dd = close / high6m - 1                                   # baisse depuis le plus haut 6 mois (négatif)
    vol = np.log(close).diff().rolling(60, min_periods=20).std()
    ampleur = -dd / (vol * math.sqrt(21))                     # baisse exprimée en « mois normaux » de variation
    return pd.DataFrame({"close": close, "high6m": high6m, "dd": dd, "ampleur": ampleur,
                         "rsi": rsi(close), "sma200": close.rolling(200, min_periods=120).mean()})


def is_dip(row, r):
    return (row.dd <= -r["baisse_min"] and row.ampleur >= r["ampleur_min"] and row.rsi <= r["rsi_max"])


def should_alert(prev, day, dd, r):
    """Anti-doublon : une alerte par creux, sauf si la baisse s'aggrave nettement."""
    if not prev:
        return True
    days = (pd.Timestamp(day) - pd.Timestamp(prev["date"])).days
    return days >= r["pause_jours"] or dd <= prev["dd"] - r["aggravation"]


def history_count(ind, r):
    """Nombre d'alertes qu'aurait envoyées le bot sur les 2 dernières années (pour calibrer)."""
    prev, n = None, 0
    for day, row in ind.iloc[-504:].iterrows():
        if is_dip(row, r) and should_alert(prev, day, row.dd, r):
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

    for a in cfg["actions"]:
        nom, t = a["nom"], a["ticker"]
        try:
            close = load_prices(t)
        except Exception as e:  # une action en panne ne doit pas bloquer les autres
            close, err = None, e
        if close is None or len(close) < 30:
            msg = f"⚠️ {nom} ({t}) : cours indisponibles, à vérifier."
            print(msg)
            rows.append(f"| {nom} | {t} | indisponible | | | | |")
            if not essai and state.get(t, {}).get("erreur") != str(today):
                alerts.append(msg)
                state.setdefault(t, {})["erreur"] = str(today)
            continue
        ind = indicators(close).dropna(subset=["dd", "ampleur", "rsi"])
        last, day = ind.iloc[-1], ind.index[-1]
        age = (today - day.date()).days
        tendance = "" if math.isnan(last.sma200) else ("au-dessus" if last.close >= last.sma200 else "sous")
        creux = is_dip(last, r)
        rows.append(f"| {nom} | {t} | {last.close:.2f} | {fmt_pct(last.dd)} | {last.ampleur:.1f} | "
                    f"{last.rsi:.0f} | {'**CREUX**' if creux else 'non'} |")
        print(f"{nom:20s} {t:9s} cours {last.close:9.2f}  vs plus haut 6 mois {fmt_pct(last.dd):>6s}  "
              f"ampleur {last.ampleur:4.1f}  RSI {last.rsi:3.0f}  moy. 200 j : {tendance or '?'}  "
              f"dernier cours {day.date()}  creux={creux}"
              + (f"  | alertes sur 2 ans : {history_count(ind, r)}" if essai else ""))
        if age > 5 and not essai:
            alerts.append(f"⚠️ {nom} ({t}) : dernier cours du {day.date()}, données figées ?")
        prev = state.get(t, {}).get("alerte")
        if creux and age <= 5 and should_alert(prev, day, last.dd, r):
            force = "très marqué" if (last.ampleur >= 2.5 or last.rsi <= 30) else "net"
            contexte = {"au-dessus": "la tendance de fond reste haussière (au-dessus de la moyenne 200 jours)",
                        "sous": "attention, l'action est aussi sous sa moyenne 200 jours : la tendance de fond "
                                "s'est retournée, vérifie s'il y a une mauvaise nouvelle",
                        "": ""}[tendance]
            msg = (f"📉 **{nom}** ({t}) : creux {force}. Cours {last.close:.2f}, {fmt_pct(last.dd)} sous son plus "
                   f"haut des 6 derniers mois ({last.high6m:.2f}), soit {last.ampleur:.1f} fois sa variation "
                   f"mensuelle habituelle, RSI {last.rsi:.0f}." + (f" {contexte[0].upper()}{contexte[1:]}." if contexte else ""))
            if prev:
                msg += f" (Déjà signalée le {prev['date']} à {fmt_pct(prev['dd'])} : la baisse s'est poursuivie.)"
            alerts.append(msg)
            state.setdefault(t, {})["alerte"] = {"date": str(day.date()), "dd": round(float(last.dd), 4)}

    if essai:
        print("\n".join(alerts) or "Aucune alerte.")
        return
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(f"# Actions surveillées\n\nDernière vérification : {datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC\n\n"
                "| Action | Ticker | Cours | vs plus haut 6 mois | Ampleur | RSI | Creux |\n|---|---|---|---|---|---|---|\n"
                + "\n".join(rows) + "\n\nUn creux est signalé quand l'action est au moins "
                f"{r['baisse_min']:.0%} sous son plus haut des 6 derniers mois, que cette baisse vaut au moins "
                f"{r['ampleur_min']} fois sa variation mensuelle habituelle, et que le RSI est sous {r['rsi_max']}.\n")
    json.dump(state, open(STATE, "w"), indent=1, ensure_ascii=False)
    if alerts:
        open(ALERTS, "w", encoding="utf-8").write("\n".join(alerts) + "\n")
    print("\n".join(alerts) or "Aucune alerte.")


if __name__ == "__main__":
    main()
