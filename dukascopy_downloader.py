from __future__ import annotations

import argparse
import csv
import gzip
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import DUKASCOPY_PRICE_SCALE, DUKASCOPY_RAW_DIR, DUKASCOPY_SYMBOL, DUKASCOPY_TICK_DIR, DUKASCOPY_YEARS, ensure_directories


BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">IIIff")


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def dukascopy_hour_url(symbol: str, day: date, hour: int) -> str:
    # Dukascopy uses zero-based month folders: January=00, December=11.
    month_folder = day.month - 1
    return f"{BASE_URL}/{symbol}/{day.year}/{month_folder:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"


def raw_hour_path(symbol: str, day: date, hour: int) -> Path:
    return DUKASCOPY_RAW_DIR / symbol / f"{day.year}" / f"{day.month:02d}" / f"{day.day:02d}" / f"{hour:02d}h_ticks.bi5"


def daily_tick_path(symbol: str, day: date) -> Path:
    return DUKASCOPY_TICK_DIR / symbol / f"{day.year}" / f"{symbol}_ticks_{day:%Y%m%d}.csv.gz"


def fetch_hour(
    symbol: str,
    day: date,
    hour: int,
    overwrite: bool = False,
    retries: int = 3,
    prefer_cache: bool = False,
) -> bytes | None:
    path = raw_hour_path(symbol, day, hour)
    if path.exists() and (prefer_cache or not overwrite):
        return path.read_bytes()

    url = dukascopy_hour_url(symbol, day, hour)
    request = Request(url, headers={"User-Agent": "AI_GOLD_RESEARCH/1.0"})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read()
            break
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(2 + attempt * 3)
                continue
            return None
        except URLError:
            if attempt < retries:
                time.sleep(2 + attempt * 3)
                continue
            return None
        except TimeoutError:
            if attempt < retries:
                time.sleep(2 + attempt * 3)
                continue
            return None
        except Exception:
            return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def decode_bi5(payload: bytes, day: date, hour: int, price_scale: int = DUKASCOPY_PRICE_SCALE) -> list[dict]:
    if not payload:
        return []
    try:
        data = lzma.decompress(payload)
    except lzma.LZMAError:
        return []

    rows = []
    hour_start = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
    for offset in range(0, len(data), RECORD.size):
        chunk = data[offset : offset + RECORD.size]
        if len(chunk) != RECORD.size:
            continue
        ms, ask_raw, bid_raw, ask_volume, bid_volume = RECORD.unpack(chunk)
        timestamp = hour_start + timedelta(milliseconds=ms)
        ask = ask_raw / price_scale
        bid = bid_raw / price_scale
        rows.append(
            {
                "time": timestamp.isoformat(),
                "bid": bid,
                "ask": ask,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "spread": ask - bid,
                "mid": (ask + bid) / 2,
            }
        )
    return rows


def download_day(
    symbol: str,
    day: date,
    overwrite: bool = False,
    pause_seconds: float = 0.0,
    workers: int = 8,
    rebuild_from_raw: bool = False,
) -> Path | None:
    output = daily_tick_path(symbol, day)
    if output.exists() and not overwrite:
        return output

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_hour, symbol, day, hour, overwrite, 3, rebuild_from_raw): hour
            for hour in range(24)
        }
        for future in as_completed(futures):
            try:
                payload = future.result()
            except Exception:
                payload = None
            if payload:
                rows.extend(decode_bi5(payload, day, futures[future]))
            if pause_seconds:
                time.sleep(pause_seconds)

    if not rows:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "bid", "ask", "bid_volume", "ask_volume", "spread", "mid"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: item["time"]))
    return output


def default_start_end(years: int = DUKASCOPY_YEARS) -> tuple[date, date]:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=365 * years)
    return start, end


def download_range(
    symbol: str,
    start: date,
    end: date,
    overwrite: bool = False,
    workers: int = 8,
    day_workers: int = 1,
    rebuild_from_raw: bool = False,
) -> list[Path]:
    ensure_directories()
    files = []
    days = list(daterange(start, end))
    if day_workers <= 1:
        for day in days:
            path = download_day(
                symbol,
                day,
                overwrite=overwrite,
                workers=workers,
                rebuild_from_raw=rebuild_from_raw,
            )
            if path:
                files.append(path)
                print(f"OK {day} -> {path}", flush=True)
            else:
                print(f"SKIP {day} no ticks", flush=True)
        return files

    with ThreadPoolExecutor(max_workers=day_workers) as executor:
        futures = {
            executor.submit(download_day, symbol, day, overwrite, 0.0, workers, rebuild_from_raw): day
            for day in days
        }
        for future in as_completed(futures):
            day = futures[future]
            path = future.result()
            if path:
                files.append(path)
                print(f"OK {day} -> {path}", flush=True)
            else:
                print(f"SKIP {day} no ticks", flush=True)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Dukascopy XAUUSD tick history as daily csv.gz files.")
    parser.add_argument("--symbol", default=DUKASCOPY_SYMBOL)
    parser.add_argument("--start", help="YYYY-MM-DD. Defaults to 5 years ago.")
    parser.add_argument("--end", help="YYYY-MM-DD. Defaults to yesterday UTC.")
    parser.add_argument("--years", type=int, default=DUKASCOPY_YEARS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-from-raw", action="store_true", help="Rewrite daily csv.gz using cached raw .bi5 files first.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--day-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start, end = default_start_end(args.years)
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    files = download_range(
        args.symbol,
        start,
        end,
        overwrite=args.overwrite,
        workers=args.workers,
        day_workers=args.day_workers,
        rebuild_from_raw=args.rebuild_from_raw,
    )
    print(f"downloaded_days={len(files)}")


if __name__ == "__main__":
    main()
