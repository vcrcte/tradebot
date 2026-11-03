"""Bot de trading crypto — exécution.

  python bot.py                 # un passage (idéal pour cron / planificateur)
  python bot.py --loop          # tourne en continu, vérifie toutes les 5 min
  python bot.py --status        # affiche capital, positions, stats
  python bot.py --live          # ordres RÉELS (exige aussi mode: live dans config.yaml)
"""
import argparse
import json
import os
import sys
import time

import ccxt
import pandas as pd
import yaml

from broker import LiveBroker, PaperBroker
from engine import Engine, new_state
from notifier import Telegram

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path=os.path.join(HERE, ".env")):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def state_path(cfg, live):
    base, ext = os.path.splitext(cfg["state_file"])
    return os.path.join(HERE, f"{base}_{'live' if live else 'paper'}{ext}")


def load_state(cfg, live=False):
    path = state_path(cfg, live)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return new_state(None if live else cfg["paper"]["initial_cash"])


def save_state(cfg, state, live=False):
    path = state_path(cfg, live)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)  # écriture atomique : pas d'état corrompu si coupure


def make_exchange(cfg, live):
    klass = getattr(ccxt, cfg["exchange"])
    params = {"enableRateLimit": True}
    if live:
        key = os.getenv("EXCHANGE_API_KEY")
        secret = os.getenv("EXCHANGE_API_SECRET")
        if not key or not secret:
            sys.exit("EXCHANGE_API_KEY / EXCHANGE_API_SECRET manquants (voir .env.example).")
        params.update(apiKey=key, secret=secret)
    return with_system_network(klass(params))


def with_system_network(ex):
    """ccxt ignore par défaut les proxys et certificats du système (trust_env=False).
    On les réactive : sans effet sur une connexion normale, indispensable derrière un proxy."""
    if getattr(ex, "session", None) is not None:
        ex.session.trust_env = True
    return ex


def closed_candles(ex, symbol, timeframe, limit=400):
    raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now = ex.milliseconds()
    df = df[df["ts"] + tf_ms <= now]  # retire la bougie en cours de formation
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def run_once(cfg, live, notify):
    state = load_state(cfg, live)
    ex = make_exchange(cfg, live)
    if live:
        broker = LiveBroker(ex, cfg["quote"], cfg["live"]["capital_cap"], state)
    else:
        broker = PaperBroker(state, cfg["paper"]["fee"], cfg["paper"]["slippage"])
    eng = Engine(cfg, state, broker, notify)
    prices, data = {}, {}
    for sym in cfg["symbols"]:  # d'abord tous les prix, pour valoriser correctement le portefeuille
        try:
            df = closed_candles(ex, sym, cfg["timeframe"])
        except ccxt.BaseError as e:
            notify(f"⚠️ {sym} : impossible de récupérer les prix ({type(e).__name__}: {e})")
            continue
        if not df.empty:
            data[sym] = df
            prices[sym] = float(df["close"].iloc[-1])
    for sym, df in data.items():
        last_ts = int(df["ts"].iloc[-1])
        if state["last_ts"].get(sym) == last_ts:
            continue  # bougie déjà traitée
        try:
            eng.on_candle(sym, df, prices)
        except ccxt.BaseError as e:
            notify(f"⚠️ {sym} : erreur d'ordre ({type(e).__name__}: {e})")
            continue
        state["last_ts"][sym] = last_ts
        save_state(cfg, state, live)
    equity = eng.equity(prices) if prices else None
    try:
        export_dashboard(cfg, state, equity, data, live)
    except Exception as e:  # le tableau de bord ne doit jamais bloquer le trading
        print(f"[dashboard] export impossible : {e}")
    save_state(cfg, state, live)
    return state, equity


def export_dashboard(cfg, state, equity, data, live):
    """Génère dashboard.html avec les données intégrées (aucune lecture de fichier : compatible Safari)."""
    now = int(time.time() * 1000)
    hist = state.setdefault("equity_history", [])
    if equity is not None:
        hist.append([now, round(equity, 2)])
        del hist[:-5000]
    payload = {
        "updated": now, "mode": "live" if live else "paper", "exchange": cfg["exchange"],
        "quote": cfg["quote"], "timeframe": cfg["timeframe"], "symbols": cfg["symbols"],
        "initial": state.get("initial_equity"), "equity": equity, "cash": state["cash"],
        "halted": state["halted"], "positions": state["positions"],
        "trades": state["trades"][-100:], "n_trades": len(state["trades"]),
        "history": hist,
        "candles": {s: [[int(t), round(float(c), 6)] for t, c in zip(df["ts"].tail(168), df["close"].tail(168))]
                    for s, df in data.items()},
    }
    tpl_path = os.path.join(HERE, "dashboard_template.html")
    if not os.path.exists(tpl_path):
        return
    blob = json.dumps(payload).replace("</", "<\\/")  # sûr à l'intérieur d'une balise <script>
    html = open(tpl_path, encoding="utf-8").read().replace("/*__BOT_DATA__*/null", blob)
    path = os.getenv("DASHBOARD_OUT") or os.path.join(HERE, "dashboard.html")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(path + ".tmp", path)


def status(cfg):
    s = load_state(cfg, cfg["mode"] == "live")
    t = s["trades"]
    wins = [x for x in t if x["pnl"] > 0]
    cash = s['cash'] if s['cash'] is not None else float('nan')
    print(f"Cash : {cash:.2f} | positions : {len(s['positions'])} | arrêté : {s['halted']}")
    for sym, p in s["positions"].items():
        print(f"  {sym}: {p['qty']:.6g} @ {p['entry']:.4f}, stop {p['stop']:.4f}")
    if t:
        print(f"Trades : {len(t)} | gagnants : {len(wins)/len(t):.0%} | "
              f"P&L total : {sum(x['pnl'] for x in t):+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    if a.status:
        return status(cfg)

    live = a.live and cfg["mode"] == "live"
    if a.live != (cfg["mode"] == "live"):
        sys.exit("Pour trader en réel il faut À LA FOIS mode: live dans config.yaml ET --live. "
                 "Sinon, lance sans --live avec mode: paper.")

    notify = Telegram()
    print(f"Mode : {'RÉEL' if live else 'simulation'} | {cfg['exchange']} | {cfg['symbols']}")
    while True:
        state, eq = run_once(cfg, live, notify)
        if eq is not None:
            print(f"Capital estimé : {eq:.2f} {cfg['quote']} | positions : {list(state['positions'])}")
        if not a.loop:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
