"""Stratégie de suivi de tendance (long uniquement, spot).

Entrée : EMA rapide > EMA lente, prix au-dessus de l'EMA de tendance et RSI pas en surachat,
         conditions apparues récemment (fenêtre `entry_window`).
Sortie : EMA rapide sous l'EMA lente OU stop suiveur basé sur l'ATR (géré par engine.py).

Tout est calculé sur des bougies CLÔTURÉES uniquement (pas de look-ahead).
"""
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema_fast"] = c.ewm(span=p["ema_fast"], adjust=False).mean()
    df["ema_slow"] = c.ewm(span=p["ema_slow"], adjust=False).mean()
    df["ema_trend"] = c.ewm(span=p["ema_trend"], adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(100)

    prev_close = c.shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def min_history(p: dict) -> int:
    return max(p["ema_trend"], p["ema_slow"]) + 5


def signal(df: pd.DataFrame, p: dict) -> dict:
    """df doit déjà contenir les indicateurs.

    Entrée : toutes les conditions (EMA rapide > EMA lente, prix > EMA de tendance, RSI < rsi_max)
    sont réunies maintenant ET sont devenues vraies il y a au plus `entry_window` bougies.
    La fenêtre rend le bot insensible à un passage sauté, sans courir après une hausse ancienne.
    Sortie : EMA rapide passée sous l'EMA lente (état, pas seulement le croisement)."""
    window = int(p.get("entry_window", 1))
    cond = (df.ema_fast > df.ema_slow) & (df.close > df.ema_trend) & (df.rsi < p["rsi_max"])
    started = cond & ~cond.shift(fill_value=False)
    enter = bool(cond.iloc[-1] and started.iloc[-window:].any())
    last = df.iloc[-1]
    return {"enter": enter, "exit": bool(last.ema_fast < last.ema_slow), "close": float(last.close),
            "atr": float(last.atr), "low": float(last.low), "open": float(last.open)}
