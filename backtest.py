"""Backtest avec EXACTEMENT le même moteur que le bot.

  python backtest.py                         # télécharge l'historique depuis l'exchange (~1 an)
  python backtest.py --days 730
  python backtest.py --csv data/BTC_EUR.csv  # CSV : ts,open,high,low,close,volume (ts en ms)
  python backtest.py --synthetic             # données aléatoires (test du code, pas de la stratégie)
"""
import argparse
import os

import numpy as np
import pandas as pd

from bot import HERE, load_cfg, with_system_network
from broker import PaperBroker
from engine import Engine, new_state


def fetch_history(cfg, symbol, days):
    import ccxt
    ex = with_system_network(getattr(ccxt, cfg["exchange"])({"enableRateLimit": True}))
    tf_ms = ex.parse_timeframe(cfg["timeframe"]) * 1000
    since = ex.milliseconds() - days * 86_400_000
    rows = []
    while since < ex.milliseconds() - tf_ms:
        batch = ex.fetch_ohlcv(symbol, cfg["timeframe"], since=since, limit=720)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if len(batch) < 2:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    expected = days * 86_400_000 // tf_ms
    if len(df) < 0.8 * expected:
        print(f"  ⚠️ seulement {len(df)} bougies reçues sur ~{expected} attendues. Kraken ne sert que les "
              f"720 dernières bougies par API : utilise --csv avec ses fichiers historiques (voir README).")
    return df


def read_csv(path):
    """Accepte un CSV avec en-tête (ts,open,high,low,close,volume) ou le format brut Kraken
    OHLCVT sans en-tête (timestamp en secondes, o, h, l, c, volume, nb trades)."""
    first = open(path).readline().split(",")[0].strip()
    if first.replace(".", "").isdigit():
        df = pd.read_csv(path, header=None).iloc[:, :6]
        df.columns = ["ts", "open", "high", "low", "close", "volume"]
    else:
        df = pd.read_csv(path)[["ts", "open", "high", "low", "close", "volume"]]
    if df["ts"].iloc[0] < 1e12:  # secondes → millisecondes
        df["ts"] = (df["ts"] * 1000).astype("int64")
    return df.sort_values("ts").reset_index(drop=True)


def synthetic(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    # marche aléatoire avec régimes de tendance, pour exercer le code
    drift = np.repeat(rng.normal(0, 0.0008, n // 400 + 1), 400)[:n]
    ret = drift + rng.normal(0, 0.01, n)
    close = 30000 * np.exp(np.cumsum(ret))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    ts = 1_600_000_000_000 + np.arange(n) * 3_600_000
    return pd.DataFrame({"ts": ts, "open": op, "high": hi, "low": lo, "close": close, "volume": 1.0})


def run(cfg, datasets: dict):
    state = new_state(cfg["paper"]["initial_cash"])
    broker = PaperBroker(state, cfg["paper"]["fee"], cfg["paper"]["slippage"])
    eng = Engine(cfg, state, broker)
    for d in datasets.values():
        d.index = pd.to_datetime(d["ts"], unit="ms", utc=True)
    all_ts = sorted(set().union(*[set(d["ts"]) for d in datasets.values()]))
    pos_idx = {s: 0 for s in datasets}
    window = 400
    curve, prices = [], {}
    for ts in all_ts:
        for sym, d in datasets.items():
            i = pos_idx[sym]
            if i < len(d) and d["ts"].iloc[i] == ts:
                prices[sym] = float(d["close"].iloc[i])
                eng.on_candle(sym, d.iloc[max(0, i - window + 1): i + 1], prices)
                pos_idx[sym] = i + 1
        curve.append((ts, eng.equity(prices)))
    eq = pd.Series([e for _, e in curve], index=pd.to_datetime([t for t, _ in curve], unit="ms", utc=True))
    return state, eq


def report(cfg, state, eq, datasets):
    t = state["trades"]
    init = cfg["paper"]["initial_cash"]
    rets = eq.pct_change().dropna()
    per_year = pd.Timedelta(days=365) / (eq.index[1] - eq.index[0]) if len(eq) > 1 else 1
    sharpe = rets.mean() / rets.std() * np.sqrt(per_year) if rets.std() > 0 else 0
    dd = (1 - eq / eq.cummax()).max()
    bh = np.mean([d["close"].iloc[-1] / d["close"].iloc[0] - 1 for d in datasets.values()])
    wins = [x for x in t if x["pnl"] > 0]
    print("\n══════ RÉSULTATS DU BACKTEST ══════")
    print(f"Période           : {eq.index[0].date()} → {eq.index[-1].date()}")
    print(f"Capital           : {init:.0f} → {eq.iloc[-1]:.2f}  ({eq.iloc[-1]/init-1:+.1%})")
    print(f"Buy & hold moyen  : {bh:+.1%}")
    print(f"Drawdown max      : {dd:.1%}")
    print(f"Sharpe (annualisé): {sharpe:.2f}")
    print(f"Trades            : {len(t)}  | gagnants : {len(wins)/len(t):.0%}" if t else "Trades : 0")
    if t:
        print(f"R moyen / trade   : {np.mean([x['r_multiple'] for x in t]):+.2f}R")
        print(f"Frais simulés     : {cfg['paper']['fee']:.2%} par ordre + slippage {cfg['paper']['slippage']:.2%}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        eq.plot(ax=ax, lw=1.2)
        ax.set_title("Courbe de capital (backtest)")
        ax.set_ylabel(cfg["quote"])
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(HERE, "backtest_equity.png")
        fig.savefig(out, dpi=120)
        print(f"Graphique         : {out}")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--csv", nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.synthetic:
        datasets = {f"SYN{i}/EUR": synthetic(seed=i) for i in range(3)}
    elif a.csv:
        datasets = {os.path.basename(p): read_csv(p) for p in a.csv}
    else:
        datasets = {}
        for s in cfg["symbols"]:
            print(f"Téléchargement {s}…")
            datasets[s] = fetch_history(cfg, s, a.days)
    state, eq = run(cfg, datasets)
    report(cfg, state, eq, datasets)


if __name__ == "__main__":
    main()
