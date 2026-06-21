from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

from config import BROKER_TIME_OFFSET_REPORT
from data_loader import DEFAULT_MT5_PATH, resolve_broker_symbol


def estimate_broker_server_offset(symbol: str = "XAUUSD", mt5_path: str = DEFAULT_MT5_PATH) -> dict:
    initialized = mt5.initialize(path=mt5_path)
    if not initialized:
        return {
            "symbol": symbol,
            "offset_hours": 0,
            "basis": "utc",
            "fresh": False,
            "error": f"MT5 initialize failed: {mt5.last_error()}",
        }

    try:
        broker_symbol = resolve_broker_symbol(symbol)
        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            return {
                "symbol": symbol,
                "broker_symbol": broker_symbol,
                "offset_hours": 0,
                "basis": "utc",
                "fresh": False,
                "error": f"No tick available: {mt5.last_error()}",
            }

        utc_now = datetime.now(timezone.utc)
        server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        age_seconds = int((utc_now - server_time).total_seconds())
        fresh = abs(age_seconds) <= 6 * 3600
        offset_hours = int(round((server_time - utc_now).total_seconds() / 3600)) if fresh else 0
        return {
            "symbol": symbol,
            "broker_symbol": broker_symbol,
            "offset_hours": offset_hours,
            "basis": "broker_server" if fresh else "utc",
            "fresh": fresh,
            "server_timestamp": server_time.isoformat(),
            "local_utc_timestamp": utc_now.isoformat(),
            "tick_age_seconds": age_seconds,
            "note": "Dukascopy files are UTC. Add offset_hours when filtering sessions by broker server time.",
        }
    finally:
        mt5.shutdown()


def save_offset_report(report: dict, path: Path = BROKER_TIME_OFFSET_REPORT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_offset_report(path: Path = BROKER_TIME_OFFSET_REPORT) -> dict:
    if not path.exists():
        return {"offset_hours": 0, "basis": "utc", "fresh": False}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    report = estimate_broker_server_offset()
    save_offset_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
