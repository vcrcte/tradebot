"""Stratégie de suivi de tendance (long uniquement, spot).

Entrée : croisement EMA rapide > EMA lente, prix au-dessus de l'EMA de tendance,
         RSI pas en surachat.
Sortie : croisement inverse OU stop suiveur basé sur l'ATR (géré par engine.py).

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
    """df doit déjà contenir les indicateurs. Regarde les 2 dernières bougies clôturées."""
    last, prev = df.iloc[-1], df.iloc[-2]
    cross_up = prev.ema_fast <= prev.ema_slow and last.ema_fast > last.ema_slow
    cross_down = prev.ema_fast >= prev.ema_slow and last.ema_fast < last.ema_slow
    enter = bool(cross_up and last.close > last.ema_trend and last.rsi < p["rsi_max"])
    return {"enter": enter, "exit": bool(cross_down), "close": float(last.close),
            "atr": float(last.atr), "low": float(last.low), "open": float(last.open)}
