from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from config import DUKASCOPY_BARS_DIR, DUKASCOPY_SYMBOL, DUKASCOPY_TICK_DIR, DUKASCOPY_YEARS, ensure_directories
from data_loader import resample_candles
from dukascopy_downloader import decode_bi5, raw_hour_path


def default_start_end(years: int = DUKASCOPY_YEARS):
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=365 * years)
    return start, end


def tick_files(symbol: str, start, end) -> list[Path]:
    root = DUKASCOPY_TICK_DIR / symbol
    files = []
    for year_dir in sorted(root.glob("*")):
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob(f"{symbol}_ticks_*.csv.gz")):
            stamp = path.stem.split("_")[-1].replace(".csv", "")
            day = datetime.strptime(stamp, "%Y%m%d").date()
            if start <= day <= end:
                files.append(path)
    return files


def load_ticks(symbol: str, start, end) -> pd.DataFrame:
    files = tick_files(symbol, start, end)
    if not files:
        raise FileNotFoundError(f"No Dukascopy tick files found for {symbol} between {start} and {end}.")
    frames = [pd.read_csv(path, parse_dates=["time"]) for path in files]
    ticks = pd.concat(frames, ignore_index=True).sort_values("time")
    return ticks


def read_tick_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"])


def ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    data = ticks.copy()
    data["time"] = pd.to_datetime(data["time"], utc=True, format="mixed").dt.tz_convert(None)
    data = data.set_index("time")
    m1 = data.resample("1min", label="left", closed="left").agg(
        {
            "mid": ["first", "max", "min", "last"],
            "bid_volume": "sum",
            "ask_volume": "sum",
            "spread": "mean",
        }
    )
    m1.columns = ["open", "high", "low", "close", "bid_volume", "ask_volume", "spread"]
    m1["tick_volume"] = m1["bid_volume"] + m1["ask_volume"]
    m1["real_volume"] = 0
    m1 = m1.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return m1[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]


def save_bars(symbol: str, bars: pd.DataFrame, timeframe: str) -> Path:
    output = DUKASCOPY_BARS_DIR / symbol / f"{symbol}_{timeframe}.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output, index=False, compression="gzip")
    return output


def raw_days(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def ticks_to_m1_from_raw_day(symbol: str, day) -> pd.DataFrame:
    m1_frames = []
    for hour in range(24):
        path = raw_hour_path(symbol, day, hour)
        if not path.exists():
            continue
        rows = decode_bi5(path.read_bytes(), day, hour)
        if not rows:
            continue
        m1_frames.append(ticks_to_m1(pd.DataFrame(rows)))
    if not m1_frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
    return pd.concat(m1_frames, ignore_index=True).sort_values("time")


def build_bars(symbol: str, start, end, timeframes: list[str]) -> dict[str, Path]:
    ensure_directories()
    m1_frames = []
    for path in tick_files(symbol, start, end):
        ticks = read_tick_file(path)
        if ticks.empty:
            continue
        m1_frames.append(ticks_to_m1(ticks))
        print(f"M1 day built: {path.name}", flush=True)
    if not m1_frames:
        raise FileNotFoundError(f"No usable tick files found for {symbol} between {start} and {end}.")
    m1 = pd.concat(m1_frames, ignore_index=True).sort_values("time")
    outputs = {"M1": save_bars(symbol, m1, "M1")}
    for timeframe in timeframes:
        if timeframe == "M1":
            continue
        outputs[timeframe] = save_bars(symbol, resample_candles(m1, timeframe), timeframe)
    return outputs


def build_bars_from_raw(symbol: str, start, end, timeframes: list[str]) -> dict[str, Path]:
    ensure_directories()
    m1_frames = []
    for day in raw_days(start, end):
        m1_day = ticks_to_m1_from_raw_day(symbol, day)
        if m1_day.empty:
            print(f"M1 raw day skipped: {day}", flush=True)
            continue
        m1_frames.append(m1_day)
        print(f"M1 raw day built: {day} rows={len(m1_day)}", flush=True)
    if not m1_frames:
        raise FileNotFoundError(f"No usable raw tick files found for {symbol} between {start} and {end}.")
    m1 = pd.concat(m1_frames, ignore_index=True).sort_values("time")
    outputs = {"M1": save_bars(symbol, m1, "M1")}
    for timeframe in timeframes:
        if timeframe == "M1":
            continue
        outputs[timeframe] = save_bars(symbol, resample_candles(m1, timeframe), timeframe)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert stored Dukascopy tick csv.gz files into M1 and strategy bars.")
    parser.add_argument("--symbol", default=DUKASCOPY_SYMBOL)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--years", type=int, default=DUKASCOPY_YEARS)
    parser.add_argument("--timeframes", default="M5,M15,H1")
    parser.add_argument("--from-raw", action="store_true", help="Build bars directly from cached raw .bi5 ticks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start, end = default_start_end(args.years)
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    if args.from_raw:
        outputs = build_bars_from_raw(args.symbol, start, end, timeframes)
    else:
        outputs = build_bars(args.symbol, start, end, timeframes)
    for timeframe, path in outputs.items():
        print(f"{timeframe}: {path}")


if __name__ == "__main__":
    main()
