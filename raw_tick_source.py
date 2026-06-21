from __future__ import annotations

from datetime import timedelta

import pandas as pd

from dukascopy_downloader import decode_bi5, raw_hour_path


def iter_days(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_raw_ticks(symbol: str, start, end) -> pd.DataFrame:
    frames = []
    for day in iter_days(start, end):
        rows = []
        for hour in range(24):
            path = raw_hour_path(symbol, day, hour)
            if not path.exists():
                continue
            rows.extend(decode_bi5(path.read_bytes(), day, hour))
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["time", "bid", "ask", "bid_volume", "ask_volume", "spread", "mid"])
    ticks = pd.concat(frames, ignore_index=True)
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True, format="mixed").dt.tz_convert(None)
    return ticks.sort_values("time").reset_index(drop=True)
