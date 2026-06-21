from __future__ import annotations

from datetime import datetime, timezone, timedelta

from config import DUKASCOPY_SYMBOL, DUKASCOPY_YEARS
from dukascopy_downloader import download_range
from tick_to_bars import build_bars


def main() -> None:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=365 * DUKASCOPY_YEARS)
    print(f"Downloading {DUKASCOPY_SYMBOL} Dukascopy ticks from {start} to {end}")
    download_range(DUKASCOPY_SYMBOL, start, end)
    outputs = build_bars(DUKASCOPY_SYMBOL, start, end, ["M5", "M15", "H1"])
    for timeframe, path in outputs.items():
        print(f"{timeframe}: {path}")


if __name__ == "__main__":
    main()
