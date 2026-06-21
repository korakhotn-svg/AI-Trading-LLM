from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DUKASCOPY_BARS_DIR, DUKASCOPY_SYMBOL


def load_dukascopy_bars(symbol: str = DUKASCOPY_SYMBOL, timeframe: str = "M5") -> pd.DataFrame:
    path = DUKASCOPY_BARS_DIR / symbol / f"{symbol}_{timeframe}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {timeframe} bars: {path}. Run dukascopy_downloader.py then tick_to_bars.py first."
        )
    return pd.read_csv(path, parse_dates=["time"])


def load_strategy_frames(symbol: str, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    return {timeframe: load_dukascopy_bars(symbol, timeframe) for timeframe in sorted(set(timeframes))}


def latest_available_bars(symbol: str = DUKASCOPY_SYMBOL) -> list[Path]:
    root = DUKASCOPY_BARS_DIR / symbol
    if not root.exists():
        return []
    return sorted(root.glob(f"{symbol}_*.csv.gz"))
