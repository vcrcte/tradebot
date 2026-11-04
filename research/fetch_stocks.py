"""Télécharge l'historique journalier (cours ajustés des dividendes) d'ETF actions, obligations et
matières premières via Yahoo Finance. Exécuté par GitHub Actions. Sortie : research/data_stocks/{TICKER}.csv

Univers volontairement composé d'ETF larges (indices, secteurs, classes d'actifs) : pas de biais du
survivant comme avec des actions individuelles choisies aujourd'hui.
"""
import os

import pandas as pd
import yfinance as yf

TICKERS = {
    # actions larges
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "EFA": "Actions dev. hors US",
    "EEM": "Actions émergentes", "VGK": "Actions Europe", "EWJ": "Actions Japon",
    # secteurs US (SPDR, depuis 1998)
    "XLK": "Technologie", "XLF": "Finance", "XLE": "Énergie", "XLV": "Santé", "XLY": "Conso. discrétionnaire",
    "XLP": "Conso. de base", "XLI": "Industrie", "XLB": "Matériaux", "XLU": "Services publics",
    # défensifs / diversifiants
    "TLT": "Oblig. US 20+ ans", "IEF": "Oblig. US 7-10 ans", "SHY": "Oblig. US 1-3 ans (quasi-cash)",
    "GLD": "Or", "DBC": "Matières premières", "VNQ": "Immobilier coté",
    # change, pour convertir en euros
    "EURUSD=X": "Taux de change EUR/USD",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_stocks")
os.makedirs(OUT, exist_ok=True)
lines = []
for t, name in TICKERS.items():
    df = yf.download(t, start="1998-01-01", auto_adjust=True, progress=False, threads=False)
    if df.empty:
        lines.append(f"{t}: indisponible")
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index.name = "date"
    fname = t.replace("=X", "")
    df.to_csv(os.path.join(OUT, f"{fname}.csv"))
    lines.append(f"{t} ({name}) : {len(df)} jours depuis {df.index[0].date()}")
    print(lines[-1])
open(os.path.join(OUT, "README.txt"), "w").write(
    "Source : Yahoo Finance (cours ajustés). Généré par research/fetch_stocks.py\n\n" + "\n".join(lines) + "\n")
