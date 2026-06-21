from __future__ import annotations

from datetime import datetime

import MetaTrader5 as mt5

from config import LOGS_DIR, SYMBOL, ensure_directories
from data_loader import CandleRequest, DEFAULT_MT5_PATH, get_recent_candles, initialize_mt5, resolve_broker_symbol, shutdown_mt5


LOG_FILE = LOGS_DIR / "mt5_test.log"
def log(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    ensure_directories()
    LOG_FILE.write_text("", encoding="utf-8")
    log("Starting MT5 verification.")
    try:
        initialize_mt5(DEFAULT_MT5_PATH)
        log("MT5 terminal initialized.")
        terminal_info = mt5.terminal_info()
        log(f"Terminal info available: {terminal_info is not None}")

        broker_symbol = resolve_broker_symbol(SYMBOL)

        log(f"Symbol available: target={SYMBOL}, broker_symbol={broker_symbol}")
        candles = get_recent_candles(CandleRequest(symbol=broker_symbol, timeframe="M5", bars=20))
        log(f"Downloaded candles: {len(candles)}")
        log("Sample candles:")
        log(candles.tail(5).to_string(index=False))
        return 0
    except Exception as exc:
        log(f"MT5 test failed: {exc}")
        return 1
    finally:
        shutdown_mt5()
        log("MT5 shutdown complete.")


if __name__ == "__main__":
    raise SystemExit(main())
