"""Modèle quantitatif multi-facteurs pour cryptos (spot, long-only, EUR).

Architecture (inspirée des pratiques des fonds systématiques) :
  1. Univers dynamique : un actif n'est éligible qu'après 365 jours de cotation et avec un volume
     suffisant, à chaque date (pas de biais d'anticipation sur les cotations).
  2. Facteurs : momentum de série temporelle multi-horizons (normalisé par la volatilité),
     momentum en coupe transversale, retournement court terme, tendance, régime de volatilité,
     choc de volume, distance au plus haut annuel, régime de marché (tendance du BTC).
  3. Prévision : ensemble d'une régression ridge et d'un gradient boosting régularisé, prédisant
     le rendement à 5 jours ajusté du risque. Entraînement walk-forward (fenêtre croissante,
     ré-entraînement trimestriel) avec purge et embargo pour éviter toute fuite d'information.
  4. Portefeuille : poids proportionnels au signal positif / volatilité, plafonds par actif,
     volatilité cible du portefeuille, bande de non-transaction pour limiter les frais.
  5. Coûts : frais + slippage appliqués à chaque euro échangé.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ANN = 365  # crypto : marché ouvert tous les jours


@dataclass
class Params:
    horizon: int = 5               # rendement futur prédit (jours)
    min_history: int = 365         # jours de cotation avant éligibilité
    min_eur_volume: float = 2e5    # volume quotidien moyen minimal (EUR, 30 j)
    retrain_every: int = 63        # ré-entraînement trimestriel
    min_train_rows: int = 1500
    embargo: int = 5               # jours retirés entre fin d'entraînement et prédiction
    target_vol: float = 0.30       # volatilité annuelle cible du portefeuille
    max_weight: float = 0.35       # poids max par actif
    max_gross: float = 1.0         # pas de levier
    band: float = 0.03             # on ne retouche une ligne que si l'écart dépasse 3 points
    cost: float = 0.0045           # 0,40 % frais taker Kraken + 0,05 % slippage
    ridge_alpha: float = 10.0
    gbm: dict = field(default_factory=lambda: dict(max_iter=200, learning_rate=0.03, max_depth=3,
                                                    min_samples_leaf=200, l2_regularization=1.0))


FEATURES = ["tsm_7", "tsm_21", "tsm_63", "tsm_126", "xs_21", "xs_63", "rev_1", "rev_3",
            "trend_100", "ema_spread", "vol_regime", "volu_shock", "dist_high", "btc_tsm_63"]


def load_panel(frames: dict) -> dict:
    """frames : {symbole: DataFrame(ts, open, high, low, close, volume)} journalier."""
    close = pd.DataFrame({s: d.set_index(pd.to_datetime(d.ts, unit="ms", utc=True).dt.normalize())["close"]
                          for s, d in frames.items()}).sort_index()
    vol = pd.DataFrame({s: d.set_index(pd.to_datetime(d.ts, unit="ms", utc=True).dt.normalize())["volume"]
                        for s, d in frames.items()}).reindex(close.index)
    close = close.asfreq("D")  # jours manquants explicites
    vol = vol.reindex(close.index)
    return {"close": close, "volume": vol}


def build_features(panel: dict, p: Params) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close, volu = panel["close"], panel["volume"]
    ret = close.pct_change(fill_method=None)
    sig = ret.ewm(span=63, min_periods=40).std()          # volatilité quotidienne
    f = {}
    for h in (7, 21, 63, 126):
        f[f"tsm_{h}"] = (close / close.shift(h) - 1) / (sig * np.sqrt(h))
    for h in (21, 63):
        f[f"xs_{h}"] = f[f"tsm_{h}"].rank(axis=1, pct=True) - 0.5
    f["rev_1"] = -ret / sig
    f["rev_3"] = -(close / close.shift(3) - 1) / (sig * np.sqrt(3))
    f["trend_100"] = (close / close.ewm(span=100, min_periods=60).mean() - 1) / (sig * np.sqrt(20))
    f["ema_spread"] = (close.ewm(span=20).mean() / close.ewm(span=50).mean() - 1) / (sig * np.sqrt(10))
    f["vol_regime"] = np.log(ret.ewm(span=21, min_periods=15).std() / ret.ewm(span=126, min_periods=60).std())
    eur_vol = volu * close
    f["volu_shock"] = np.log((eur_vol.rolling(7).mean() + 1) / (eur_vol.rolling(63).mean() + 1))
    f["dist_high"] = close / close.rolling(365, min_periods=200).max() - 1
    btc = f["tsm_63"]["BTC"] if "BTC" in close else f["tsm_63"].mean(axis=1)
    f["btc_tsm_63"] = pd.DataFrame({s: btc for s in close.columns})

    # univers éligible à chaque date (info disponible à la date uniquement)
    listed_days = close.notna().cumsum()
    liquid = eur_vol.rolling(30, min_periods=20).mean() > p.min_eur_volume
    eligible = (listed_days >= p.min_history) & liquid & close.notna()

    # cible : rendement des `horizon` prochains jours, ajusté du risque, démoyenné à chaque date
    fwd = close.shift(-p.horizon) / close - 1
    target = fwd / (sig * np.sqrt(p.horizon))

    long = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    long["target"] = target.stack(future_stack=True)
    long["eligible"] = eligible.stack(future_stack=True)
    long = long[long["eligible"] == True].drop(columns="eligible")  # noqa: E712
    long = long.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    # normalisation en coupe transversale par date (robuste : écrêtage à ±3)
    def zs(g):
        z = (g - g.mean()) / (g.std(ddof=0) + 1e-9)
        return z.clip(-3, 3)
    xs_cols = [c for c in FEATURES if c != "btc_tsm_63"]
    long[xs_cols] = long.groupby(level=0)[xs_cols].transform(zs)
    long["btc_tsm_63"] = long["btc_tsm_63"].clip(-3, 3)
    long["target"] = long["target"].clip(-4, 4)
    return long, ret, sig


def walk_forward_predict(long: pd.DataFrame, p: Params, start=None, end=None) -> pd.Series:
    """Prédictions strictement hors échantillon. Chaque modèle n'utilise que des lignes dont la
    cible était entièrement connue avant la date de prédiction (purge = horizon + embargo)."""
    dates = long.index.get_level_values(0).unique().sort_values()
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end, tz="UTC")]
    preds, i = [], 0
    while i < len(dates):
        t0 = dates[i]
        cutoff = t0 - pd.Timedelta(days=p.horizon + p.embargo)
        train = long.loc[long.index.get_level_values(0) <= cutoff].dropna(subset=["target"])
        block = dates[i:i + p.retrain_every]
        test = long.loc[long.index.get_level_values(0).isin(block)]
        if len(train) >= p.min_train_rows and len(test):
            X, y = train[FEATURES].values, train["target"].values
            ridge = Ridge(alpha=p.ridge_alpha).fit(X, y)
            gbm = HistGradientBoostingRegressor(random_state=0, **p.gbm).fit(X, y)
            pr = 0.5 * ridge.predict(test[FEATURES].values) + 0.5 * gbm.predict(test[FEATURES].values)
            preds.append(pd.Series(pr, index=test.index))
        i += p.retrain_every
    return pd.concat(preds).sort_index() if preds else pd.Series(dtype=float)


def target_weights(score: pd.DataFrame, ret: pd.DataFrame, sig: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Poids cibles à la clôture de t (exécutés au prix de clôture de t, rendement à partir de t+1)."""
    raw = score.clip(lower=0) / sig.reindex_like(score)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).clip(upper=p.max_weight)
    out = []
    r = ret.reindex(columns=w.columns).fillna(0)
    for t in w.index:
        wt = w.loc[t].values
        if wt.sum() <= 0:
            out.append(wt)
            continue
        hist = r.loc[:t].tail(90)
        cov = hist.ewm(span=45).cov().loc[hist.index[-1]].values * ANN
        pv = float(np.sqrt(max(wt @ cov @ wt, 1e-12)))
        scale = min(p.target_vol / pv, p.max_gross / wt.sum())
        out.append(wt * scale)
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def simulate(weights: pd.DataFrame, ret: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Applique la bande de non-transaction, les coûts et la dérive des poids."""
    r = ret.reindex(index=weights.index, columns=weights.columns).fillna(0)
    r_next = r.shift(-1).fillna(0)
    cur = np.zeros(weights.shape[1])
    rows = []
    for t, tgt in zip(weights.index, weights.values):
        diff = tgt - cur
        trade = np.where(np.abs(diff) > p.band, diff, 0.0)
        if tgt.sum() == 0:              # sortie complète : on vend tout
            trade = -cur
        cur = cur + trade
        turnover = np.abs(trade).sum()
        cost = turnover * p.cost
        nxt = r_next.loc[t].values  # rendement du jour suivant
        gross = float(cur @ nxt)
        net = gross - cost
        # dérive des poids avec les prix
        growth = 1 + net
        cur = cur * (1 + nxt) / growth if growth > 0 else cur * 0
        rows.append((t, net, gross, turnover, cost, float(np.abs(cur).sum())))
    df = pd.DataFrame(rows, columns=["date", "net", "gross", "turnover", "cost", "exposure"]).set_index("date")
    return df.iloc[:-1]  # le dernier jour n'a pas de rendement suivant


def metrics(r: pd.Series, name: str = "") -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(ANN)
    sharpe = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
    down = r[r < 0].std() * np.sqrt(ANN)
    dd = (1 - eq / eq.cummax()).max()
    return {"stratégie": name, "CAGR": cagr, "volatilité": vol, "Sharpe": sharpe,
            "Sortino": r.mean() * ANN / down if down > 0 else np.nan, "drawdown max": dd,
            "Calmar": cagr / dd if dd > 0 else np.nan, "total": eq.iloc[-1] - 1}


def block_bootstrap_sharpe(r: pd.Series, block: int = 20, n: int = 2000, seed: int = 0) -> tuple:
    """Intervalle de confiance à 90 % du Sharpe par bootstrap par blocs (préserve l'autocorrélation)."""
    r = r.dropna().values
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(r) / block))
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, len(r) - block, k)])[:len(r)]
        x = r[idx]
        out.append(x.mean() / x.std() * np.sqrt(ANN))
    return float(np.percentile(out, 5)), float(np.percentile(out, 95)), float(np.mean(np.array(out) > 0))


def information_coefficient(pred: pd.Series, long: pd.DataFrame) -> pd.Series:
    """Corrélation de rang quotidienne entre prévision et rendement futur réalisé."""
    df = pd.DataFrame({"pred": pred, "target": long["target"].reindex(pred.index)}).dropna()
    return df.groupby(level=0).apply(lambda g: g["pred"].corr(g["target"], method="spearman") if len(g) >= 4 else np.nan)
