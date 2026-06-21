from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import DUKASCOPY_SYMBOL, RESULTS_DIR, ensure_directories
from data_source import load_strategy_frames
from raw_tick_source import load_raw_ticks
from strategy_schema import load_strategy
from tick_backtest_engine import daily_returns, generate_signals, simulate_tick_trades, summarize_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confirm a selected strategy with raw Dukascopy tick execution.")
    parser.add_argument("--strategy", default="strategies/target_loop_best_strategy.json")
    parser.add_argument("--symbol", default=DUKASCOPY_SYMBOL)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-06-19")
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--label", default="tick_confirm_forward_2026")
    return parser.parse_args()


def as_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def chunk_ranges(start, end, chunk_days: int):
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def main() -> None:
    args = parse_args()
    ensure_directories()
    start, end = as_date(args.start), as_date(args.end)
    strategy_path = Path(args.strategy)
    if not strategy_path.is_absolute():
        strategy_path = Path.cwd() / strategy_path
    strategy = load_strategy(strategy_path)

    frames = load_strategy_frames(args.symbol, [strategy.entry_timeframe, strategy.confirmation_timeframe, strategy.trend_timeframe])
    signals = generate_signals(frames, strategy)
    signals = signals[(signals["time"].dt.date >= start) & (signals["time"].dt.date <= end)]

    all_trades = []
    chunk_rows = []
    for chunk_start, chunk_end in chunk_ranges(start, end, args.chunk_days):
        tick_end = min(chunk_end + timedelta(days=1), end)
        chunk_signals = signals[(signals["time"].dt.date >= chunk_start) & (signals["time"].dt.date <= chunk_end)]
        print(f"chunk {chunk_start}..{chunk_end} signals={len(chunk_signals)}", flush=True)
        if chunk_signals.empty:
            continue
        ticks = load_raw_ticks(args.symbol, chunk_start, tick_end)
        print(f"  ticks={len(ticks)}", flush=True)
        trades = simulate_tick_trades(ticks, chunk_signals, strategy)
        if trades.empty:
            chunk_rows.append({"start": str(chunk_start), "end": str(chunk_end), "signals": len(chunk_signals), "trades": 0})
            continue
        all_trades.append(trades)
        chunk_summary = summarize_trades(trades, strategy, start=chunk_start, end=chunk_end)
        chunk_rows.append({"start": str(chunk_start), "end": str(chunk_end), "signals": len(chunk_signals), **chunk_summary})
        print(
            f"  trades={chunk_summary['trades']} return={chunk_summary['total_return_pct']:.2f}% "
            f"dd={chunk_summary['max_drawdown_pct']:.2f}% avg_day={chunk_summary['avg_trade_day_return_pct']:.2f}%",
            flush=True,
        )

    trades = pd.concat(all_trades, ignore_index=True).sort_values("entry_time") if all_trades else pd.DataFrame()
    summary = summarize_trades(trades, strategy, start=start, end=end)
    daily = daily_returns(trades, strategy)

    prefix = f"{strategy.name}_{args.label}"
    trades_path = RESULTS_DIR / f"{prefix}_trades.csv"
    daily_path = RESULTS_DIR / f"{prefix}_daily_returns.csv"
    summary_path = RESULTS_DIR / f"{prefix}_summary.json"
    chunks_path = RESULTS_DIR / f"{prefix}_chunks.csv"

    if not trades.empty:
        trades.to_csv(trades_path, index=False)
        daily.to_csv(daily_path, index=False)
    pd.DataFrame(chunk_rows).to_csv(chunks_path, index=False)
    summary_path.write_text(
        json.dumps({**summary, "strategy_params": asdict(strategy)}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({**summary, "summary_file": str(summary_path), "daily_file": str(daily_path), "trades_file": str(trades_path), "chunks_file": str(chunks_path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
