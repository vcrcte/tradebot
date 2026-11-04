"""Recherche actions (ETF) : stratégies déclarées AVANT tout test.

S1  Allocation tactique par tendance (type Faber) : SPY, EFA, EEM, IEF, TLT, GLD, DBC, VNQ à poids
    égaux ; chaque ligne n'est détenue que si son cours > moyenne 210 jours (~10 mois), sinon sa
    part va en obligations courtes (SHY).
S2  Double momentum : parmi SPY, QQQ, IWM, EFA, EEM, VGK, EWJ, GLD, DBC, VNQ, les 3 meilleurs
    momentums 12-1 mois, à condition de battre le rendement 12 mois de SHY ; parts non attribuées
    en obligations 7-10 ans (IEF).
S3  Rotation sectorielle filtrée : les 3 meilleurs secteurs US sur 6 mois (9 SPDR historiques)
    quand le S&P 500 est au-dessus de sa moyenne 200 jours ; sinon 100 % IEF.
Références : S&P 500 (SPY) détenu, 60/40 SPY/IEF rééquilibré chaque mois.

Tout en EUR (rendements USD convertis au taux EUR/USD : le risque de change est inclus).
Rééquilibrage au dernier jour ouvré du mois, exécuté à la clôture, rendements à partir du lendemain.
Coûts : 0,10 % par euro échangé (commission IBKR ~0,05 % + glissement 0,05 %), uniquement les jours
de rééquilibrage.

Protocole : sélection sur 2004-2019 (meilleur Sharpe), test unique sur 2020 → aujourd'hui.
Règle de décision fixée à l'avance : simulation seulement si, sur le test, Sharpe ≥ SPY ET
pire baisse < SPY.

  python research/stocks.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_model as qm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_stocks")
OUT = os.path.join(HERE, "results")
DEV = ("2004-01-01", "2019-12-31")
TEST = ("2020-01-01", None)
COST = 0.0010
ANN = 252

S1_ASSETS = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"]
S2_ASSETS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "GLD", "DBC", "VNQ"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]


def load():
    px = {}
    for f in glob.glob(os.path.join(DATA, "*.csv")):
        t = os.path.basename(f)[:-4]
        px[t] = pd.read_csv(f, index_col="date", parse_dates=True)["Close"]
    px = pd.DataFrame(px).sort_index()
    fx = px.pop("EURUSD").ffill()                       # dollars pour 1 euro
    days = px["SPY"].dropna().index                     # jours de bourse US
    px = px.reindex(days).ffill(limit=5)
    fx = fx.reindex(days).ffill()
    ret_usd = px.pct_change(fill_method=None)
    ret_eur = (1 + ret_usd).mul(fx.shift(1) / fx, axis=0) - 1    # un actif en USD vu par un investisseur en EUR
    return px, ret_eur


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return set(s.groupby([idx.year, idx.month]).max())


def simulate(targets: dict, ret: pd.DataFrame, start, end) -> pd.DataFrame:
    """targets : {date de rééquilibrage: Series de poids}. Trading uniquement ces jours-là."""
    cols = sorted({c for w in targets.values() for c in w.index})
    r = ret[cols].fillna(0)
    idx = r.loc[start:end].index
    cur = pd.Series(0.0, index=cols)
    out = []
    for i, t in enumerate(idx[:-1]):
        cost = 0.0
        if t in targets:
            tgt = targets[t].reindex(cols).fillna(0)
            cost = float((tgt - cur).abs().sum()) * COST
            cur = tgt.copy()
        nxt = r.loc[idx[i + 1]]
        gross = float((cur * nxt).sum())
        net = gross - cost
        cur = cur * (1 + nxt) / (1 + gross) if (1 + gross) > 0 else cur * 0
        out.append((idx[i + 1], net, cost, t in targets))
    return pd.DataFrame(out, columns=["date", "net", "cost", "rebal"]).set_index("date")


def s1(px, dates):
    tg = {}
    sma = px.rolling(210, min_periods=200).mean()
    for t in dates:
        avail = [a for a in S1_ASSETS if pd.notna(sma.at[t, a])]
        if not avail:
            continue
        w = pd.Series(0.0, index=S1_ASSETS + ["SHY"])
        for a in avail:
            w[a if px.at[t, a] > sma.at[t, a] else "SHY"] += 1 / len(avail)
        tg[t] = w
    return tg


def s2(px, dates):
    tg = {}
    for t in dates:
        hist = px.loc[:t]
        if len(hist) < 260:
            continue
        p12, p1, p0 = hist.iloc[-253], hist.iloc[-22], hist.iloc[-1]
        mom = (p1 / p12 - 1)[S2_ASSETS].dropna()
        cash12 = p0["SHY"] / p12["SHY"] - 1
        if len(mom) < 3:
            continue
        top = [a for a in mom.sort_values(ascending=False).index[:3] if mom[a] > cash12]
        w = pd.Series(0.0, index=S2_ASSETS + ["IEF"])
        for a in top:
            w[a] = 1 / 3
        w["IEF"] = 1 - w.sum()
        tg[t] = w
    return tg


def s3(px, dates):
    tg = {}
    sma200 = px["SPY"].rolling(200).mean()
    for t in dates:
        hist = px.loc[:t]
        if len(hist) < 130 or pd.isna(sma200.at[t]):
            continue
        w = pd.Series(0.0, index=SECTORS + ["IEF"])
        if px.at[t, "SPY"] > sma200.at[t]:
            mom = (hist.iloc[-1] / hist.iloc[-127] - 1)[SECTORS].dropna()
            for a in mom.sort_values(ascending=False).index[:3]:
                w[a] = 1 / 3
        else:
            w["IEF"] = 1.0
        tg[t] = w
    return tg


def bench_spy(dates):
    return {t: pd.Series({"SPY": 1.0}) for t in sorted(dates)[:1]}


def bench_6040(dates):
    return {t: pd.Series({"SPY": 0.6, "IEF": 0.4}) for t in dates}


def metrics(r, name):
    eq = (1 + r).cumprod()
    yrs = len(r) / ANN
    dd = (1 - eq / eq.cummax()).max()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(ANN)
    return {"stratégie": name, "CAGR": cagr, "volatilité": r.std() * np.sqrt(ANN),
            "Sharpe": r.mean() / r.std() * np.sqrt(ANN), "Sortino": r.mean() * ANN / downside,
            "pire baisse": dd, "Calmar": cagr / dd, "total": eq.iloc[-1] - 1}


def run(period):
    px, ret = load()
    start, end = period
    idx = px.loc[start:end].index
    me = sorted(d for d in month_ends(px.index) if d >= idx[0] - pd.Timedelta(days=31) and (end is None or d <= pd.Timestamp(end)))
    me_set = set(me)
    strats = {
        "S1 Allocation tactique (tendance)": s1(px, me),
        "S2 Double momentum": s2(px, me),
        "S3 Rotation sectorielle filtrée": s3(px, me),
        "S&P 500 détenu (SPY)": {idx[0]: pd.Series({"SPY": 1.0})},
        "60/40 actions/obligations": bench_6040(me),
    }
    rets, extra = {}, {}
    for name, tg in strats.items():
        # premier rééquilibrage = premier jour de la période (on part investi)
        first = max((d for d in tg if d <= idx[0]), default=None)
        tg2 = {d: w for d, w in tg.items() if d >= idx[0]}
        if first is not None:
            tg2[idx[0]] = tg[first]
        sim = simulate(tg2, ret, idx[0], end)
        rets[name] = sim["net"]
        extra[name] = {"frais/an": sim["cost"].sum() / (len(sim) / ANN),
                       "rotation/an": sim["cost"].sum() / COST / (len(sim) / ANN)}
    rets = pd.DataFrame(rets).dropna()
    table = pd.DataFrame([{**metrics(rets[c], c), **extra[c]} for c in rets]).set_index("stratégie")
    yearly = rets.groupby(rets.index.year).apply(lambda g: (1 + g).prod() - 1)
    return table, yearly, rets, strats


def show(table, yearly, title):
    fmt = table.copy()
    for c in ["CAGR", "total"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:+.1%}")
    for c in ["volatilité", "pire baisse", "frais/an"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2%}" if c == "frais/an" else f"{x:.1%}")
    for c in ["Sharpe", "Sortino", "Calmar", "rotation/an"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2f}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"\n===== {title} =====")
    print(fmt.to_string())
    print("\nPar année :\n" + yearly.map(lambda x: f"{x:+.1%}").to_string())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t_dev, y_dev, r_dev, _ = run(DEV)
    show(t_dev, y_dev, f"DÉVELOPPEMENT {DEV[0]} → {DEV[1]} (en EUR, frais inclus)")
    cands = [i for i in t_dev.index if i.startswith(("S1", "S2", "S3"))]
    chosen = t_dev.loc[cands, "Sharpe"].idxmax()
    print(f"\n>>> Stratégie retenue (meilleur Sharpe en développement) : {chosen}")
    t_test, y_test, r_test, strats = run(TEST)
    show(t_test, y_test, f"TEST UNIQUE {TEST[0]} → aujourd'hui (en EUR, frais inclus)")
    spy = t_test.loc["S&P 500 détenu (SPY)"]
    c = t_test.loc[chosen]
    ok = c["Sharpe"] >= spy["Sharpe"] and c["pire baisse"] < spy["pire baisse"]
    print(f"\n>>> Règle de décision pour « {chosen} » : Sharpe {c['Sharpe']:.2f} vs SPY {spy['Sharpe']:.2f} ; "
          f"pire baisse {c['pire baisse']:.1%} vs SPY {spy['pire baisse']:.1%} → {'VALIDÉE' if ok else 'REFUSÉE'}")
    lo, hi, _ = qm.block_bootstrap_sharpe(r_test[chosen])
    print(f"    Intervalle de confiance 90 % du Sharpe (test, base 365 j) : [{lo * np.sqrt(ANN / 365):.2f} ; {hi * np.sqrt(ANN / 365):.2f}]")
    last = max(strats[chosen])
    w = strats[chosen][last]
    print(f"    Positions actuelles ({last.date()}) : " + ", ".join(f"{k} {v:.0%}" for k, v in w[w > 0].items()))
    t_dev.to_csv(os.path.join(OUT, "stocks_dev_metrics.csv"))
    t_test.to_csv(os.path.join(OUT, "stocks_test_metrics.csv"))
    y_test.to_csv(os.path.join(OUT, "stocks_test_yearly.csv"))
