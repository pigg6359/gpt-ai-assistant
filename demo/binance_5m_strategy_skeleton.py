"""
Binance 5m Strategy Skeleton (BTC/SOL/BNB)
- Volatility filter (ATR%)
- Regime filter (ADX + EMA)
- Trend breakout OR range mean-reversion
- Risk-based position sizing (R)
NOTE: This is a skeleton. You must adapt for your account, fees, and execution rules.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import ccxt
import pandas as pd

# -----------------------------
# Config
# -----------------------------
@dataclass
class SymbolConfig:
    symbol: str
    atrp_min: float  # ATR% threshold
    max_spread_bp: float = 8.0  # max spread in basis points (0.08%)


CFG = {
    "timeframe": "5m",
    "lookback": 300,  # candles to fetch
    "risk_per_trade": 0.005,  # 0.5% equity risk per trade
    "max_positions": 2,
    "cooldown_bars": 3,  # bars to wait after closing / entering
    "adx_trend": 20,
    "ema_fast": 20,
    "ema_slow": 50,
    "atr_len": 14,
    "adx_len": 14,
    "rsi_len": 14,
    "bb_len": 20,
    "bb_k": 2,
    "breakout_len": 20,
    "sl_atr_mult": 1.5,
    "tp_r_mult": 2.0,
    "time_stop_bars": 12,  # ~60 minutes
}


SYMBOLS = [
    SymbolConfig("BTC/USDT", atrp_min=0.0012),
    SymbolConfig("SOL/USDT", atrp_min=0.0020),
    SymbolConfig("BNB/USDT", atrp_min=0.0015),
]

# -----------------------------
# Exchange init (spot or futures)
# -----------------------------

def make_exchange() -> ccxt.Exchange:
    ex = ccxt.binance({
        "enableRateLimit": True,
        # "apiKey": "...",
        # "secret": "...",
        # Futures example:
        # "options": {"defaultType": "future"},
    })
    return ex

# -----------------------------
# Indicators
# -----------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def bollinger(close: pd.Series, n: int, k: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    # Simple ADX implementation
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = atr(df, 1)  # true range per bar
    atr_n = tr.rolling(n).mean()

    plus_di = 100 * (plus_dm.rolling(n).mean() / (atr_n + 1e-12))
    minus_di = 100 * (minus_dm.rolling(n).mean() / (atr_n + 1e-12))
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    return dx.rolling(n).mean()

# -----------------------------
# Data
# -----------------------------

def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def get_spread_bp(ex: ccxt.Exchange, symbol: str) -> float:
    ob = ex.fetch_order_book(symbol, limit=5)
    bid = ob["bids"][0][0] if ob["bids"] else None
    ask = ob["asks"][0][0] if ob["asks"] else None
    if not bid or not ask:
        return 9999.0
    mid = (bid + ask) / 2
    return (ask - bid) / mid * 10000

# -----------------------------
# Signal
# -----------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_f"] = ema(df["close"], CFG["ema_fast"])
    df["ema_s"] = ema(df["close"], CFG["ema_slow"])
    df["atr"] = atr(df, CFG["atr_len"])
    df["atrp"] = df["atr"] / df["close"]
    df["rsi"] = rsi(df["close"], CFG["rsi_len"])
    df["adx"] = adx(df, CFG["adx_len"])
    ma, bbu, bbl = bollinger(df["close"], CFG["bb_len"], CFG["bb_k"])
    df["bb_ma"], df["bb_u"], df["bb_l"] = ma, bbu, bbl
    df["hhv"] = df["high"].rolling(CFG["breakout_len"]).max()
    df["llv"] = df["low"].rolling(CFG["breakout_len"]).min()
    return df


def decide_signal(df: pd.DataFrame, sym_cfg: SymbolConfig) -> Tuple[str, Optional[str]]:
    """
    Returns (regime, action)
    action: "LONG", "SHORT", or None
    """
    last = df.iloc[-1]
    # 1) volatility filter
    if float(last["atrp"]) < sym_cfg.atrp_min:
        return "quiet", None

    # 2) regime
    is_trend = float(last["adx"]) >= CFG["adx_trend"]
    bull = float(last["ema_f"]) > float(last["ema_s"])
    bear = float(last["ema_f"]) < float(last["ema_s"])

    # 3) signals
    if is_trend:
        # breakout
        if bull and float(last["close"]) > float(last["hhv"]):
            return "trend", "LONG"
        if bear and float(last["close"]) < float(last["llv"]):
            return "trend", "SHORT"
        return "trend", None

    # mean reversion
    if float(last["rsi"]) < 28 and float(last["close"]) < float(last["bb_l"]):
        return "range", "LONG"
    if float(last["rsi"]) > 72 and float(last["close"]) > float(last["bb_u"]):
        return "range", "SHORT"
    return "range", None

# -----------------------------
# Risk / position sizing
# -----------------------------

def position_size_usdt(equity_usdt: float, entry: float, sl: float) -> float:
    # risk = equity * risk_per_trade
    risk = equity_usdt * CFG["risk_per_trade"]
    per_unit_loss = abs(entry - sl)
    if per_unit_loss <= 0:
        return 0.0
    qty = risk / per_unit_loss
    # qty is in "base asset" units (BTC/SOL/BNB)
    return qty


def compute_sl_tp(side: str, entry: float, atr_val: float) -> Tuple[float, float]:
    if side == "LONG":
        sl = entry - CFG["sl_atr_mult"] * atr_val
        tp = entry + CFG["tp_r_mult"] * (entry - sl)
    else:
        sl = entry + CFG["sl_atr_mult"] * atr_val
        tp = entry - CFG["tp_r_mult"] * (sl - entry)
    return sl, tp

# -----------------------------
# Execution placeholders
# -----------------------------

def place_order_stub(ex: ccxt.Exchange, symbol: str, side: str, qty: float) -> None:
    # Replace with create_order(...) using your preferred order type.
    print(f"[ORDER] {symbol} {side} qty={qty:.6f}")

# -----------------------------
# State
# -----------------------------

class State:
    def __init__(self) -> None:
        self.positions: Dict[str, Dict] = {}  # symbol -> {side, entry, sl, tp, entry_ts, bars_in_trade}
        self.cooldown: Dict[str, int] = {}  # symbol -> remaining bars cooldown

    def in_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def can_trade(self, symbol: str) -> bool:
        return (self.cooldown.get(symbol, 0) <= 0) and (len(self.positions) < CFG["max_positions"])

# -----------------------------
# Main loop
# -----------------------------

def main() -> None:
    ex = make_exchange()
    st = State()

    # TODO: replace with real equity retrieval from balance
    equity_usdt = 1000.0

    while True:
        loop_start = time.time()

        for sym_cfg in SYMBOLS:
            symbol = sym_cfg.symbol

            # cooldown tick
            if st.cooldown.get(symbol, 0) > 0:
                st.cooldown[symbol] -= 1

            # basic spread check
            try:
                spread_bp = get_spread_bp(ex, symbol)
            except Exception as exc:
                print(f"[WARN] orderbook fail {symbol}: {exc}")
                continue

            if spread_bp > sym_cfg.max_spread_bp:
                print(f"[SKIP] {symbol} spread too wide: {spread_bp:.2f}bp")
                continue

            # fetch & features
            try:
                df = fetch_ohlcv(ex, symbol, CFG["timeframe"], CFG["lookback"])
            except Exception as exc:
                print(f"[WARN] fetch_ohlcv fail {symbol}: {exc}")
                continue

            df = compute_features(df).dropna()
            if len(df) < 60:
                continue

            last = df.iloc[-1]
            price = float(last["close"])
            atr_val = float(last["atr"])

            # manage existing position (skeleton)
            if st.in_position(symbol):
                pos = st.positions[symbol]
                pos["bars_in_trade"] += 1

                # time stop example
                if pos["bars_in_trade"] >= CFG["time_stop_bars"]:
                    print(f"[EXIT] {symbol} time-stop")
                    del st.positions[symbol]
                    st.cooldown[symbol] = CFG["cooldown_bars"]
                # TODO: add stop-loss / take-profit checks using current price or exchange position PnL
                continue

            # entry
            if not st.can_trade(symbol):
                continue

            regime, action = decide_signal(df, sym_cfg)
            if not action:
                continue

            sl, tp = compute_sl_tp(action, price, atr_val)
            qty = position_size_usdt(equity_usdt, price, sl)
            if qty <= 0:
                continue

            # place order (placeholder)
            place_order_stub(ex, symbol, action, qty)

            # record state
            st.positions[symbol] = {
                "side": action,
                "entry": price,
                "sl": sl,
                "tp": tp,
                "entry_ts": df.iloc[-1]["ts"],
                "bars_in_trade": 0,
                "regime": regime,
                "atrp": float(last["atrp"]),
            }
            st.cooldown[symbol] = CFG["cooldown_bars"]

            print(
                f"[ENTER] {symbol} {action} entry={price:.2f} SL={sl:.2f} "
                f"TP={tp:.2f} regime={regime}"
            )

        # align to 5-min loop (simple)
        elapsed = time.time() - loop_start
        sleep_s = max(1.0, 300.0 - elapsed)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
