"""Moteur commun au bot (paper/live) et au backtest : une seule logique, testée une fois.

Gestion du risque :
- risque par trade = demi-Kelly estimé sur l'historique des trades, borné [risk_min, risk_max]
  (risk_default tant qu'il y a moins de `kelly_min_trades` trades)
- stop initial et suiveur à atr_mult x ATR
- allocation max par position, nb max de positions
- coupe-circuit journalier (plus de nouvelles entrées) et drawdown max (arrêt total)
"""
from datetime import datetime, timezone

import strategy


def new_state(initial_cash) -> dict:
    """initial_cash=None en réel : les références de capital sont fixées au premier passage."""
    return {"cash": initial_cash, "initial_equity": initial_cash, "peak_equity": initial_cash,
            "positions": {}, "trades": [], "last_ts": {}, "day": None,
            "day_start_equity": initial_cash, "halted": False}


def kelly_risk(trades: list, cfg: dict) -> float:
    r = cfg["risk"]
    if len(trades) < r["kelly_min_trades"]:
        return r["risk_default"]
    rs = [t["r_multiple"] for t in trades[-100:]]
    wins = [x for x in rs if x > 0]
    losses = [-x for x in rs if x <= 0]
    if not wins or not losses:
        return r["risk_default"]
    w = len(wins) / len(rs)
    payoff = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
    kelly = w - (1 - w) / payoff
    return max(r["risk_min"], min(r["risk_max"], 0.5 * kelly))


class Engine:
    def __init__(self, cfg: dict, state: dict, broker, notify=lambda msg: None):
        self.cfg, self.s, self.b, self.notify = cfg, state, broker, notify
        self.p = cfg["strategy"]

    def equity(self, prices: dict) -> float:
        pos_val = sum(p["qty"] * prices.get(sym, p["entry"]) for sym, p in self.s["positions"].items())
        return self.b.cash() + pos_val

    def check_risk(self, equity: float, ts_ms: int):
        r = self.cfg["risk"]
        if self.s["peak_equity"] is None:
            self.s["initial_equity"] = self.s["peak_equity"] = equity
        day = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()
        if self.s["day"] != day:
            self.s["day"], self.s["day_start_equity"] = day, equity
        self.s["peak_equity"] = max(self.s["peak_equity"], equity)
        dd = 1 - equity / self.s["peak_equity"]
        if not self.s["halted"] and dd >= r["max_drawdown"]:
            self.s["halted"] = True
            self.notify(f"🛑 ARRÊT : drawdown {dd:.1%} ≥ {r['max_drawdown']:.0%}. "
                        f"Le bot n'ouvre plus rien. Remets halted=false dans state.json pour relancer.")
        daily_loss = 1 - equity / self.s["day_start_equity"]
        return daily_loss < r["max_daily_loss"] and not self.s["halted"]

    def on_candle(self, symbol: str, df, prices: dict):
        """df = bougies clôturées (OHLCV) jusqu'à la dernière incluse."""
        if len(df) < strategy.min_history(self.p):
            return
        d = strategy.add_indicators(df, self.p)
        sig = strategy.signal(d, self.p)
        ts = int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], "timestamp") else int(df.index[-1])
        prices[symbol] = sig["close"]
        can_enter = self.check_risk(self.equity(prices), ts)
        pos = self.s["positions"].get(symbol)
        mult = self.p["atr_mult"]

        if pos:
            if sig["low"] <= pos["stop"]:
                # stop touché ; si la bougie a ouvert sous le stop (gap), on sort à l'ouverture
                self._exit(symbol, min(pos["stop"], sig["open"]), "stop", ts)
            elif sig["exit"]:
                self._exit(symbol, sig["close"], "signal", ts)
            else:
                pos["stop"] = max(pos["stop"], sig["close"] - mult * sig["atr"])
            return

        if sig["enter"] and can_enter and len(self.s["positions"]) < self.cfg["risk"]["max_positions"]:
            stop = sig["close"] - mult * sig["atr"]
            if stop <= 0:
                return
            equity = self.equity(prices)
            risk = kelly_risk(self.s["trades"], self.cfg)
            qty = equity * risk / (sig["close"] - stop)
            qty = min(qty, equity * self.cfg["risk"]["max_alloc"] / sig["close"])
            fill = self.b.buy(symbol, qty, sig["close"])
            if fill:
                self.s["positions"][symbol] = {"qty": fill["qty"], "entry": fill["price"], "stop": stop,
                                               "init_risk": fill["price"] - stop, "cost": fill["cost"],
                                               "opened": ts}
                self.notify(f"🟢 ACHAT {symbol} {fill['qty']:.6g} @ {fill['price']:.4f} "
                            f"| stop {stop:.4f} | risque {risk:.1%}")

    def _exit(self, symbol, price, reason, ts):
        pos = self.s["positions"][symbol]
        fill = self.b.sell(symbol, pos["qty"], price)
        if not fill:
            return
        pnl = fill["proceeds"] - pos["cost"]
        r_mult = (fill["price"] - pos["entry"]) / pos["init_risk"] if pos["init_risk"] > 0 else 0
        self.s["trades"].append({"symbol": symbol, "entry": pos["entry"], "exit": fill["price"],
                                 "pnl": pnl, "r_multiple": r_mult, "reason": reason,
                                 "opened": pos["opened"], "closed": ts})
        del self.s["positions"][symbol]
        self.notify(f"{'✅' if pnl > 0 else '🔴'} VENTE {symbol} @ {fill['price']:.4f} ({reason}) "
                    f"| P&L {pnl:+.2f} | {r_mult:+.2f}R")
