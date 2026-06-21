from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from config import SYMBOL


DEFAULT_MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}

RESAMPLE_RULES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
}


@dataclass(frozen=True)
class CandleRequest:
    symbol: str = SYMBOL
    timeframe: str = "M1"
    bars: int = 20000


def initialize_mt5(path: str | None = None) -> None:
    launch_path = path or DEFAULT_MT5_PATH
    initialized = mt5.initialize(path=launch_path)
    if not initialized:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    mt5.shutdown()


def symbol_available(symbol: str = SYMBOL) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        return False
    return bool(info.visible or mt5.symbol_select(symbol, True))


def resolve_broker_symbol(symbol: str = SYMBOL) -> str:
    if symbol_available(symbol):
        return symbol

    symbols = mt5.symbols_get()
    if not symbols:
        raise RuntimeError("No symbols returned by MT5. Check broker login and Market Watch.")

    target = symbol.upper()
    candidates = [item.name for item in symbols if item.name.upper().startswith(target)]
    if not candidates and target == "XAUUSD":
        candidates = [item.name for item in symbols if "XAU" in item.name.upper() or "GOLD" in item.name.upper()]

    for candidate in candidates:
        if symbol_available(candidate):
            return candidate

    raise RuntimeError(f"{symbol} is not available and no broker alias could be selected.")


def get_recent_candles(request: CandleRequest) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(
        request.symbol,
        TIMEFRAME_MAP[request.timeframe],
        0,
        request.bars,
    )
    if rates is None:
        raise RuntimeError(f"Could not load candles: {mt5.last_error()}")
    candles = pd.DataFrame(rates)
    if candles.empty:
        raise RuntimeError("MT5 returned no candles.")
    candles["time"] = pd.to_datetime(candles["time"], unit="s")
    return candles


def get_m1_history(symbol: str, bars: int = 20000) -> pd.DataFrame:
    return get_recent_candles(CandleRequest(symbol=symbol, timeframe="M1", bars=bars))


def resample_candles(m1_candles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe not in RESAMPLE_RULES:
        raise ValueError(f"Unsupported timeframe for resampling: {timeframe}")
    if timeframe == "M1":
        return m1_candles.copy()

    data = m1_candles.copy().sort_values("time").set_index("time")
    resampled = data.resample(RESAMPLE_RULES[timeframe], label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
            "spread": "last",
            "real_volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return resampled


def build_timeframe_data(m1_candles: pd.DataFrame, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    return {timeframe: resample_candles(m1_candles, timeframe) for timeframe in sorted(set(timeframes))}


def estimate_server_time_offset(symbol: str) -> dict:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {
            "symbol": symbol,
            "offset_hours": 0,
            "fresh": False,
            "note": f"No tick available: {mt5.last_error()}",
        }

    utc_now = datetime.now(timezone.utc)
    server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    age_seconds = (utc_now - server_time).total_seconds()
    raw_offset_hours = round((server_time - utc_now).total_seconds() / 3600)
    fresh = abs(age_seconds) <= 6 * 3600

    return {
        "symbol": symbol,
        "server_timestamp": server_time.isoformat(),
        "local_utc_timestamp": utc_now.isoformat(),
        "offset_hours": int(raw_offset_hours) if fresh else 0,
        "fresh": fresh,
        "tick_age_seconds": int(age_seconds),
        "note": "Offset applied only when latest tick is fresh; otherwise UTC is used.",
    }


def save_server_time_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
