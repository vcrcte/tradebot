"""Protocole de recherche.

  python research/run_research.py dev       # période de développement (jusqu'au 31/10/2025)
  python research/run_research.py holdout   # test final unique sur la dernière année, jamais vue

Sorties : research/results/{phase}_*.csv / .png / .json
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_model as qm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_END = "2025-10-31"
HOLD_START = "2025-11-01"
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def load(tf="1d"):
    frames = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", f"*_{tf}.csv"))):
        sym = os.path.basename(f).split("_")[0]
        frames[sym] = pd.read_csv(f)
    return frames


def baseline_buy_hold(ret, sym, idx):
    return ret[sym].reindex(idx).shift(-1).fillna(0).iloc[:-1].rename(f"Buy & hold {sym}")


def baseline_equal_weight(ret, eligible, idx, p):
    """Équipondéré sur l'univers éligible, rebalancé le 1er de chaque mois, avec coûts."""
    w = eligible.reindex(idx).fillna(False).astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    monthly = w.where(w.index.to_series().dt.is_month_start, np.nan).ffill().fillna(0)
    pp = qm.Params(**{**p.__dict__, "band": 0.0})
    return qm.simulate(monthly, ret, pp)["net"].rename("Équipondéré (mensuel)")


def baseline_tsmom(close, ret, sig, eligible, idx, p):
    """Momentum simple : 63 j normalisé par la vol, long si positif. Même construction de portefeuille."""
    score = ((close / close.shift(63) - 1) / (sig * np.sqrt(63))).where(eligible).reindex(idx)
    w = qm.target_weights(score.fillna(0), ret, sig, p)
    return qm.simulate(w, ret, p)["net"].rename("Momentum simple (TSMOM)")


def current_bot(frames4h, start, end):
    """Stratégie actuelle du bot (EMA + fenêtre, 4 h) sur BTC/ETH/SOL, rendements journaliers."""
    sys.path.insert(0, os.path.dirname(HERE))
    import backtest
    import bot
    cfg = bot.load_cfg(os.path.join(os.path.dirname(HERE), "config.yaml"))
    ds = {}
    for s in ("BTC", "ETH", "SOL"):
        d = frames4h[s].copy()
        t = pd.to_datetime(d.ts, unit="ms", utc=True)
        d = d[(t >= pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=60)) & (t <= pd.Timestamp(end, tz="UTC"))]
        ds[f"{s}/EUR"] = d.reset_index(drop=True)
    st, eq = backtest.run(cfg, ds)
    daily = eq.resample("D").last().ffill()
    return daily.pct_change().loc[start:end].rename("Bot actuel (EMA 4 h)"), st


def by_year(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.groupby(rets.index.year).apply(lambda g: (1 + g).prod() - 1)


def main(phase):
    t0 = time.time()
    p = qm.Params()
    frames = load("1d")
    print(f"{len(frames)} actifs : {', '.join(frames)}")
    panel = qm.load_panel(frames)
    long, ret, sig = qm.build_features(panel, p)
    close = panel["close"]
    eur_vol = panel["volume"] * close
    eligible = (close.notna().cumsum() >= p.min_history) & (eur_vol.rolling(30, min_periods=20).mean() > p.min_eur_volume)

    if phase == "dev":
        start, end = None, DEV_END
    else:
        start, end = HOLD_START, None
    pred = qm.walk_forward_predict(long, p, start=start, end=end)
    print(f"prédictions hors échantillon : {len(pred)} lignes, {pred.index.get_level_values(0).nunique()} jours "
          f"({time.time() - t0:.0f} s)")
    score = pred.unstack().reindex(columns=close.columns)
    idx = score.index
    w = qm.target_weights(score.fillna(0), ret, sig, p)
    sim = qm.simulate(w, ret, p)

    rets = pd.DataFrame({"Modèle quant": sim["net"]})
    rets["Modèle quant (avant frais)"] = sim["gross"]
    rets[baseline_buy_hold(ret, "BTC", idx).name] = baseline_buy_hold(ret, "BTC", idx)
    eqw = baseline_equal_weight(ret, eligible, idx, p)
    rets[eqw.name] = eqw
    ts = baseline_tsmom(close, ret, sig, eligible, idx, p)
    rets[ts.name] = ts
    try:
        cb, _ = current_bot(load("4h"), str(idx[0].date()), str(idx[-1].date()))
        rets[cb.name] = cb.reindex(rets.index).fillna(0)
    except Exception as e:  # noqa: BLE001
        print(f"bot actuel non comparé : {e}")
    rets = rets.dropna(how="all").fillna(0)

    table = pd.DataFrame([qm.metrics(rets[c], c) for c in rets.columns]).set_index("stratégie")
    lo, hi, pos = qm.block_bootstrap_sharpe(rets["Modèle quant"])
    ic = qm.information_coefficient(pred, long)
    yearly = by_year(rets)
    extra = {
        "période": f"{rets.index[0].date()} → {rets.index[-1].date()}",
        "jours": len(rets),
        "sharpe_IC90": [round(lo, 2), round(hi, 2)],
        "proba_sharpe_positif_bootstrap": round(pos, 3),
        "IC_moyen": round(float(ic.mean()), 4),
        "IC_t_stat": round(float(ic.mean() / ic.std() * np.sqrt(ic.count())), 2),
        "IC_par_année": {int(k): round(float(v), 4) for k, v in ic.groupby(ic.index.year).mean().items()},
        "rotation_annuelle": round(float(sim["turnover"].mean() * qm.ANN), 2),
        "frais_annuels": round(float(sim["cost"].mean() * qm.ANN), 4),
        "exposition_moyenne": round(float(sim["exposure"].mean()), 3),
        "poids_moyens": {k: round(float(v), 3) for k, v in w.mean().sort_values(ascending=False).items() if v > 0.005},
    }
    pd.set_option("display.width", 200)
    fmt = table.copy()
    for c in ["CAGR", "volatilité", "drawdown max", "total"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:+.1%}" if c in ("CAGR", "total") else f"{x:.1%}")
    for c in ["Sharpe", "Sortino", "Calmar"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2f}")
    print("\n" + fmt.to_string())
    print("\nRendement par année :\n" + yearly.map(lambda x: f"{x:+.1%}").to_string())
    print("\n" + json.dumps(extra, ensure_ascii=False, indent=1))

    table.to_csv(os.path.join(OUT, f"{phase}_metrics.csv"))
    yearly.to_csv(os.path.join(OUT, f"{phase}_yearly.csv"))
    rets.to_csv(os.path.join(OUT, f"{phase}_daily_returns.csv"))
    json.dump(extra, open(os.path.join(OUT, f"{phase}_stats.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nterminé en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
