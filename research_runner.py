from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DUKASCOPY_SYMBOL, RESULTS_DIR, STRATEGIES_DIR, ensure_directories
from strategy_schema import default_strategy, load_strategy, save_strategy
from tick_backtest_engine import run_tick_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python VPS runner: strategy -> tick backtest -> result files.")
    parser.add_argument("--symbol", default=DUKASCOPY_SYMBOL)
    parser.add_argument("--strategy-file")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    strategy = load_strategy(Path(args.strategy_file)) if args.strategy_file else default_strategy()
    strategy_path = STRATEGIES_DIR / f"{strategy.name}.json"
    save_strategy(strategy, strategy_path)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    result = run_tick_backtest(strategy, start, end, symbol=args.symbol)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = RESULTS_DIR / "latest_tick_backtest.csv"
    pd.DataFrame([result]).to_csv(latest_path, index=False)
    print(result)
    print(f"latest={latest_path}")


if __name__ == "__main__":
    main()
