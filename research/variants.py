"""Variantes déclarées AVANT d'être testées (3 essais, comptés dans la correction pour tests multiples).

V0  Modèle ML d'origine (ridge + boosting), rééquilibrage quotidien, bande 3 %      [déjà testé]
V1  Même modèle, prévisions lissées (EMA 10 j), rééquilibrage hebdomadaire, bande 5 %
V2  Tendance multi-horizons (sans ML) : moyenne des momentums 21/63/126 j normalisés
    + écart à la moyenne 100 j, long si positif, poids ∝ signal / vol, volatilité cible,
    rééquilibrage hebdomadaire, bande 5 %
V3  V2 + filtre de régime : les cryptos autres que le BTC ne sont détenues que si la tendance
    63 j du BTC est positive

  python research/variants.py dev | holdout
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_model as qm  # noqa: E402
import run_research as rr  # noqa: E402

OUT = rr.OUT


def weekly(score: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les décisions du lundi, maintenues toute la semaine."""
    mon = score.index.dayofweek == 0
    return score.where(pd.Series(mon, index=score.index), np.nan).ffill()


def trend_score(close, sig):
    parts = [(close / close.shift(h) - 1) / (sig * np.sqrt(h)) for h in (21, 63, 126)]
    parts.append((close / close.ewm(span=100, min_periods=60).mean() - 1) / (sig * np.sqrt(20)))
    return sum(p.clip(-3, 3) for p in parts) / len(parts)


def deflated_sharpe_prob(r: pd.Series, n_trials: int, sr_trials: list) -> float:
    """Probabilité que le vrai Sharpe soit > 0 en tenant compte du nombre d'essais
    (Bailey & López de Prado, 2014), à partir des Sharpe quotidiens."""
    from scipy.stats import norm
    x = r.dropna().values
    T = len(x)
    sr = x.mean() / x.std()
    skew = ((x - x.mean()) ** 3).mean() / x.std() ** 3
    kurt = ((x - x.mean()) ** 4).mean() / x.std() ** 4
    var_sr = np.var(sr_trials) if len(sr_trials) > 1 else 0.0
    emc = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf(z))


def main(phase):
    p = qm.Params()
    pw = replace(p, band=0.05)
    frames = rr.load("1d")
    panel = qm.load_panel(frames)
    close = panel["close"]
    long, ret, sig = qm.build_features(panel, p)
    eur_vol = panel["volume"] * close
    eligible = (close.notna().cumsum() >= p.min_history) & (eur_vol.rolling(30, min_periods=20).mean() > p.min_eur_volume)
    start, end = (None, rr.DEV_END) if phase == "dev" else (rr.HOLD_START, None)

    cache = os.path.join(OUT, f"{phase}_ml_pred.pkl")
    if os.path.exists(cache):
        pred = pd.read_pickle(cache)
    else:
        pred = qm.walk_forward_predict(long, p, start=start, end=end)
        pred.to_pickle(cache)
    ml = pred.unstack().reindex(columns=close.columns)
    idx = ml.index

    res = {}
    w0 = qm.target_weights(ml.fillna(0), ret, sig, p)
    res["V0 ML quotidien"] = qm.simulate(w0, ret, p)
    ml_s = weekly(ml.ewm(span=10, min_periods=1).mean())
    res["V1 ML lissé hebdo"] = qm.simulate(qm.target_weights(ml_s.fillna(0), ret, sig, pw), ret, pw)
    tr = trend_score(close, sig).where(eligible).reindex(idx)
    res["V2 Tendance multi-horizons"] = qm.simulate(qm.target_weights(weekly(tr).fillna(0), ret, sig, pw), ret, pw)
    btc_up = (close["BTC"] / close["BTC"].shift(63) - 1).reindex(idx) > 0
    tr3 = tr.copy()
    alts = [c for c in tr3.columns if c != "BTC"]
    tr3.loc[~btc_up, alts] = np.nan
    res["V3 Tendance + régime BTC"] = qm.simulate(qm.target_weights(weekly(tr3).fillna(0), ret, sig, pw), ret, pw)

    rets = pd.DataFrame({k: v["net"] for k, v in res.items()})
    rets["Buy & hold BTC"] = rr.baseline_buy_hold(ret, "BTC", idx)
    eqw = rr.baseline_equal_weight(ret, eligible, idx, p)
    rets[eqw.name] = eqw
    try:
        cb, _ = rr.current_bot(rr.load("4h"), str(idx[0].date()), str(idx[-1].date()))
        rets[cb.name] = cb.reindex(rets.index).fillna(0)
    except Exception as e:  # noqa: BLE001
        print(f"bot actuel non comparé : {e}")
    rets = rets.dropna(how="all").fillna(0)

    variants = [k for k in res]
    daily_sr = [rets[k].mean() / rets[k].std() for k in variants]
    rows = []
    for c in rets.columns:
        m = qm.metrics(rets[c], c)
        if c in res:
            m["rotation/an"] = res[c]["turnover"].mean() * qm.ANN
            m["frais/an"] = res[c]["cost"].mean() * qm.ANN
            m["exposition"] = res[c]["exposure"].mean()
            lo, hi, _ = qm.block_bootstrap_sharpe(rets[c])
            m["Sharpe IC90"] = f"[{lo:.2f} ; {hi:.2f}]"
            m["P(Sharpe>0) corrigée"] = deflated_sharpe_prob(rets[c], len(variants), daily_sr)
        rows.append(m)
    table = pd.DataFrame(rows).set_index("stratégie")
    yearly = rr.by_year(rets)

    fmt = table.copy()
    for c in ["CAGR", "total"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:+.1%}")
    for c in ["volatilité", "drawdown max", "frais/an", "exposition"]:
        fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    for c in ["Sharpe", "Sortino", "Calmar", "rotation/an", "P(Sharpe>0) corrigée"]:
        fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    fmt = fmt.fillna("")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"Période {rets.index[0].date()} → {rets.index[-1].date()} ({len(rets)} jours)\n")
    print(fmt.to_string())
    print("\nRendement par année :\n" + yearly.map(lambda x: f"{x:+.1%}").to_string())
    table.to_csv(os.path.join(OUT, f"{phase}_variants_metrics.csv"))
    yearly.to_csv(os.path.join(OUT, f"{phase}_variants_yearly.csv"))
    rets.to_csv(os.path.join(OUT, f"{phase}_variants_daily.csv"))
    for k in ("V2 Tendance multi-horizons", "V3 Tendance + régime BTC"):
        w = qm.target_weights((weekly(tr) if k.startswith("V2") else weekly(tr3)).fillna(0), ret, sig, pw)
        print(f"\nPoids moyens {k} : " + json.dumps({c: round(float(v), 3) for c, v in w.mean().sort_values(ascending=False).items() if v > 0.005}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
