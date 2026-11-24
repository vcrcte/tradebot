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


def indicators(close, r):
    high6m = close.rolling(126, min_periods=20).max()
    dd = close / high6m - 1                                   # baisse depuis le plus haut 6 mois (négatif)
    vol_m = np.log(close).diff().rolling(60, min_periods=20).std() * math.sqrt(21)  # variation mensuelle normale
    ampleur = -dd / vol_m                                     # baisse exprimée en « mois normaux » de variation
    rs = rsi(close)
    creux = (dd <= -r["baisse_min"]) & (ampleur >= r["ampleur_min"]) & (rs <= r["rsi_max"])
    w = r["fenetre_creux"]
    low = close.rolling(w, min_periods=1).min()
    rebond = close / low - 1
    seuil_rebond = np.maximum(r["rebond_min"], 0.5 * vol_m)
    signal = (creux.astype(int).rolling(w, min_periods=1).max() == 1) & (rebond >= seuil_rebond) \
        & (dd <= -r["encore_sous"])
    return pd.DataFrame({"close": close, "high6m": high6m, "dd": dd, "ampleur": ampleur, "rsi": rs,
                         "creux": creux, "low": low, "rebond": rebond, "signal": signal,
                         "sma200": close.rolling(200, min_periods=120).mean()})


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

    for a in cfg["actions"]:
        nom, t = a["nom"], a["ticker"]
        if a.get("achete"):
            rows.append(f"| {nom} | {t} | achetée | | | | |")
            print(f"{nom:20s} {t:9s} déjà achetée : plus surveillée")
            continue
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
        ind = indicators(close, r).dropna(subset=["dd", "ampleur", "rsi"])
        last, day = ind.iloc[-1], ind.index[-1]
        age = (today - day.date()).days
        tendance = "" if math.isnan(last.sma200) else ("au-dessus" if last.close >= last.sma200 else "sous")
        etat = ("**ACHAT**" if last.signal else "creux, attend un rebond" if ind.creux.iloc[-r["fenetre_creux"]:].any()
                else "non")
        rows.append(f"| {nom} | {t} | {last.close:.2f} | {fmt_pct(last.dd)} | {last.ampleur:.1f} | "
                    f"{last.rsi:.0f} | {etat} |")
        print(f"{nom:20s} {t:9s} cours {last.close:9.2f}  vs plus haut 6 mois {fmt_pct(last.dd):>6s}  "
              f"ampleur {last.ampleur:4.1f}  RSI {last.rsi:3.0f}  moy. 200 j : {tendance or '?'}  "
              f"rebond {fmt_pct(last.rebond)}  dernier cours {day.date()}  état : {etat}"
              + (f"  | alertes sur 2 ans : {history_count(ind, r)}" if essai else ""))
        if age > 5 and not essai:
            alerts.append(f"⚠️ {nom} ({t}) : dernier cours du {day.date()}, données figées ?")
        prev = state.get(t, {}).get("alerte")
        if last.signal and age <= 5 and should_alert(prev, day, last.dd, r):
            recent = ind.iloc[-r["fenetre_creux"]:]
            force = "très marqué" if (recent.ampleur.max() >= 2.5 or recent.rsi.min() <= 30) else "net"
            contexte = {"au-dessus": "la tendance de fond reste haussière (au-dessus de la moyenne 200 jours)",
                        "sous": "attention, l'action est aussi sous sa moyenne 200 jours : la tendance de fond "
                                "s'est retournée, vérifie s'il y a une mauvaise nouvelle",
                        "": ""}[tendance]
            msg = (f"📉 **{nom}** ({t}) : creux {force} qui commence à se retourner. Cours {last.close:.2f}, "
                   f"{fmt_pct(last.dd)} sous son plus haut des 6 derniers mois ({last.high6m:.2f}), "
                   f"{fmt_pct(last.rebond)} depuis son point bas récent ({last.low:.2f}), RSI {last.rsi:.0f}." + (f" {contexte[0].upper()}{contexte[1:]}." if contexte else ""))
            if prev:
                msg += f" (Déjà signalée le {prev['date']} à {fmt_pct(prev['dd'])} : la baisse s'est poursuivie.)"
            alerts.append(msg)
            state.setdefault(t, {})["alerte"] = {"date": str(day.date()), "dd": round(float(last.dd), 4)}

    if essai:
        print("\n".join(alerts) or "Aucune alerte.")
        return
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(f"# Actions surveillées\n\nDernière vérification : {datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC\n\n"
                "| Action | Ticker | Cours | vs plus haut 6 mois | Ampleur | RSI | État |\n|---|---|---|---|---|---|---|\n"
                + "\n".join(rows) + "\n\nAchat unique : une alerte part quand l'action a fait un vrai creux "
                f"(au moins {r['baisse_min']:.0%} sous son plus haut des 6 derniers mois, au moins {r['ampleur_min']} "
                f"fois sa variation mensuelle habituelle, RSI sous {r['rsi_max']}) dans les {r['fenetre_creux']} "
                f"dernières séances, puis remonte d'au moins {r['rebond_min']:.0%} depuis son point bas tout en "
                f"restant {r['encore_sous']:.0%} sous son plus haut.\n")
    json.dump(state, open(STATE, "w"), indent=1, ensure_ascii=False)
    if alerts:
        open(ALERTS, "w", encoding="utf-8").write("\n".join(alerts) + "\n")
    print("\n".join(alerts) or "Aucune alerte.")


if __name__ == "__main__":
    main()
