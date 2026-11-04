"""Télécharge l'historique OHLCV en EUR depuis l'API publique Bitstamp (exchange européen régulé).

Exécuté par GitHub Actions (.github/workflows/research-data.yml) : les runners ont un accès
Internet complet. Sortie : research/data/{SYMBOL}_{1d|4h}.csv  (ts en ms, open, high, low, close, volume).
"""
import os
import sys
import time

import pandas as pd
import requests

PAIRS = ["btc", "eth", "sol", "xrp", "ada", "link", "dot", "ltc", "avax", "doge", "bch", "uni", "aave", "xlm"]
STEPS = {"1d": 86400, "4h": 14400}
START = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp())
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "tradebot-research"


def fetch(pair: str, step: int) -> pd.DataFrame:
    rows, start, now = [], START, int(time.time())
    while start < now - step:
        for attempt in range(4):
            try:
                r = S.get(f"https://www.bitstamp.net/api/v2/ohlc/{pair}eur/",
                          params={"step": step, "limit": 1000, "start": start}, timeout=30)
                if r.status_code == 404:
                    return pd.DataFrame()
                r.raise_for_status()
                batch = r.json().get("data", {}).get("ohlc", [])
                break
            except (requests.RequestException, ValueError) as e:
                print(f"  {pair} retry {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"{pair}: échec répété")
        if not batch:
            start += step * 1000  # trou avant la cotation : on avance
            continue
        rows += batch
        last = int(batch[-1]["timestamp"])
        if last + step <= start:
            break
        start = last + step
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).astype(float)
    df["ts"] = (df["timestamp"] * 1000).astype("int64")
    df = df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts").sort_values("ts")
    return df[df["volume"] > 0].reset_index(drop=True)


summary = []
for p in PAIRS:
    for name, step in STEPS.items():
        df = fetch(p, step)
        if df.empty:
            print(f"{p.upper()}/EUR {name}: indisponible")
            continue
        df.to_csv(os.path.join(OUT, f"{p.upper()}_{name}.csv"), index=False)
        first = pd.to_datetime(df.ts.iloc[0], unit="ms").date()
        summary.append(f"{p.upper()}/EUR {name}: {len(df)} bougies depuis {first}")
        print(summary[-1])
open(os.path.join(OUT, "README.txt"), "w").write(
    "Source : API publique Bitstamp (paires EUR). Généré par research/fetch_data.py\n\n" + "\n".join(summary) + "\n")
