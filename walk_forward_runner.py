from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DUKASCOPY_SYMBOL, RESULTS_DIR, ensure_directories
from strategy_schema import default_strategy, load_strategy
from tick_backtest_engine import run_tick_backtest
from server_time import estimate_broker_server_offset, save_offset_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtest and forward test with spread-aware tick fills.")
    parser.add_argument("--symbol", default=DUKASCOPY_SYMBOL)
    parser.add_argument("--strategy-file")
    parser.add_argument("--backtest-start", default="2025-01-01")
    parser.add_argument("--backtest-end", default="2025-09-30")
    parser.add_argument("--forward-start", default="2025-10-01")
    parser.add_argument("--forward-end", default="2025-12-31")
    parser.add_argument("--session-time-basis", choices=["utc", "broker_server"], default=None)
    parser.add_argument("--server-time-offset-hours", default="auto")
    return parser.parse_args()


def as_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    args = parse_args()
    ensure_directories()
    strategy = load_strategy(Path(args.strategy_file)) if args.strategy_file else default_strategy()
    if args.session_time_basis:
        strategy.session_time_basis = args.session_time_basis
    if args.server_time_offset_hours == "auto":
        offset_report = estimate_broker_server_offset(strategy.symbol)
        save_offset_report(offset_report)
        strategy.server_time_offset_hours = int(offset_report.get("offset_hours", 0))
    else:
        strategy.server_time_offset_hours = int(args.server_time_offset_hours)
    backtest = run_tick_backtest(
        strategy,
        as_date(args.backtest_start),
        as_date(args.backtest_end),
        symbol=args.symbol,
        label="backtest",
    )
    forward = run_tick_backtest(
        strategy,
        as_date(args.forward_start),
        as_date(args.forward_end),
        symbol=args.symbol,
        label="forward",
    )

    report = pd.DataFrame(
        [
            {"phase": "backtest", **backtest},
            {"phase": "forward", **forward},
        ]
    )
    output = RESULTS_DIR / f"{strategy.name}_walk_forward_report.csv"
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    print(f"report={output}")


if __name__ == "__main__":
    main()
