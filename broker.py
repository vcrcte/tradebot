"""Deux brokers avec la même interface :
- PaperBroker : simulation (frais + slippage), aucun ordre réel.
- LiveBroker  : ordres au marché réels via ccxt.
"""
import time


class PaperBroker:
    live = False

    def __init__(self, state: dict, fee: float, slippage: float):
        self.state, self.fee, self.slip = state, fee, slippage

    def cash(self) -> float:
        return self.state["cash"]

    def buy(self, symbol: str, qty: float, price: float):
        px = price * (1 + self.slip)
        cost = qty * px * (1 + self.fee)
        if cost > self.state["cash"]:
            qty = self.state["cash"] / (px * (1 + self.fee))
            cost = self.state["cash"]
        if qty <= 0:
            return None
        self.state["cash"] -= cost
        return {"qty": qty, "price": px, "cost": cost}

    def sell(self, symbol: str, qty: float, price: float):
        px = price * (1 - self.slip)
        proceeds = qty * px * (1 - self.fee)
        self.state["cash"] += proceeds
        return {"qty": qty, "price": px, "proceeds": proceeds}


class LiveBroker:
    live = True

    def __init__(self, exchange, quote: str, capital_cap: float, state: dict):
        self.ex, self.quote, self.cap, self.state = exchange, quote, capital_cap, state
        self.ex.load_markets()

    def cash(self) -> float:
        """Budget du bot = plafond + P&L réalisé - capital engagé, borné par le solde réel.
        Ainsi les pertes réduisent vraiment son capital et déclenchent les coupe-circuits."""
        bal = self.ex.fetch_balance()
        free = float(bal.get("free", {}).get(self.quote, 0) or 0)
        realized = sum(t["pnl"] for t in self.state["trades"])
        engaged = sum(p["cost"] for p in self.state["positions"].values())
        return max(0.0, min(free, self.cap + realized - engaged))

    def _fill(self, order, symbol):
        # Certains exchanges ne renvoient pas tout de suite le remplissage.
        for _ in range(5):
            if order.get("filled") and order.get("average"):
                break
            time.sleep(1)
            order = self.ex.fetch_order(order["id"], symbol)
        return float(order["filled"]), float(order["average"] or order["price"])

    def buy(self, symbol: str, qty: float, price: float):
        m = self.ex.market(symbol)
        qty = float(self.ex.amount_to_precision(symbol, qty))
        min_amt = (m.get("limits", {}).get("amount", {}) or {}).get("min") or 0
        min_cost = (m.get("limits", {}).get("cost", {}) or {}).get("min") or 0
        if qty <= 0 or qty < min_amt or qty * price < min_cost:
            return None
        o = self.ex.create_market_buy_order(symbol, qty)
        filled, avg = self._fill(o, symbol)
        return {"qty": filled, "price": avg, "cost": filled * avg, "id": o["id"]}

    def sell(self, symbol: str, qty: float, price: float):
        base = self.ex.market(symbol)["base"]
        free = float(self.ex.fetch_balance().get("free", {}).get(base, 0) or 0)
        qty = float(self.ex.amount_to_precision(symbol, min(qty, free)))
        if qty <= 0:
            return None
        o = self.ex.create_market_sell_order(symbol, qty)
        filled, avg = self._fill(o, symbol)
        return {"qty": filled, "price": avg, "proceeds": filled * avg, "id": o["id"]}
