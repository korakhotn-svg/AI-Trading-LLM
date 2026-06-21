from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from config import RESULTS_DIR
from strategy_schema import ResearchStrategy
from tick_backtest_engine import generate_signals, daily_returns, summarize_trades


def _slice_m1(m1: pd.DataFrame, start, end) -> pd.DataFrame:
    data = m1.copy()
    data["time"] = pd.to_datetime(data["time"], format="mixed")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    return data[(data["time"] >= start_ts) & (data["time"] < end_ts)].sort_values("time").reset_index(drop=True)


def simulate_m1_trades(m1: pd.DataFrame, signals: pd.DataFrame, strategy: ResearchStrategy, start, end) -> pd.DataFrame:
    if signals.empty or m1.empty:
        return pd.DataFrame()

    bars = _slice_m1(m1, start, end)
    if bars.empty:
        return pd.DataFrame()

    times = bars["time"].to_numpy()
    opens = bars["open"].to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    spreads = bars["spread"].fillna(bars["spread"].median()).to_numpy(dtype=float)

    signals = signals.copy()
    signals["time"] = pd.to_datetime(signals["time"], format="mixed")
    signals = signals[(signals["time"] >= pd.Timestamp(start)) & (signals["time"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if signals.empty:
        return pd.DataFrame()

    trades = []
    daily_counts: dict[str, int] = {}
    open_until = pd.Timestamp.min

    for signal in signals.itertuples(index=False):
        signal_time = pd.Timestamp(getattr(signal, "execution_time", signal.time))
        if strategy.one_position_at_time and signal_time <= open_until:
            continue

        entry_idx = int(np.searchsorted(times, np.datetime64(signal_time), side="left"))
        if entry_idx >= len(times):
            continue

        trade_day = pd.Timestamp(times[entry_idx]).date().isoformat()
        if daily_counts.get(trade_day, 0) >= strategy.max_trades_per_day:
            continue

        spread = max(float(spreads[entry_idx]), 0.0)
        half_spread = spread / 2.0
        direction = int(signal.signal)
        atr = float(getattr(signal, "atr", 0) or 0)
        if atr <= 0:
            continue

        if direction > 0:
            entry_price = float(opens[entry_idx] + half_spread)
            stop_price = entry_price - atr * strategy.atr_stop_multiplier
            target_price = entry_price + (entry_price - stop_price) * strategy.reward_risk
        else:
            entry_price = float(opens[entry_idx] - half_spread)
            stop_price = entry_price + atr * strategy.atr_stop_multiplier
            target_price = entry_price - (stop_price - entry_price) * strategy.reward_risk

        max_exit_time = pd.Timestamp(times[entry_idx]) + pd.Timedelta(hours=strategy.max_holding_hours)
        exit_idx = entry_idx
        exit_price = float(closes[entry_idx])
        exit_reason = "time"

        for idx in range(entry_idx, len(times)):
            bar_time = pd.Timestamp(times[idx])
            if bar_time > max_exit_time:
                exit_idx = idx
                exit_price = float(closes[idx])
                exit_reason = "time"
                break

            half = max(float(spreads[idx]), 0.0) / 2.0
            if direction > 0:
                stop_hit = float(lows[idx] - half) <= stop_price
                target_hit = float(highs[idx] - half) >= target_price
                if stop_hit:
                    exit_idx = idx
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if target_hit:
                    exit_idx = idx
                    exit_price = target_price
                    exit_reason = "target"
                    break
            else:
                stop_hit = float(highs[idx] + half) >= stop_price
                target_hit = float(lows[idx] + half) <= target_price
                if stop_hit:
                    exit_idx = idx
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if target_hit:
                    exit_idx = idx
                    exit_price = target_price
                    exit_reason = "target"
                    break

        risk_points = abs(entry_price - stop_price)
        pnl_points = (exit_price - entry_price) * direction
        r_multiple = pnl_points / risk_points if risk_points else 0.0
        exit_time = pd.Timestamp(times[exit_idx])
        open_until = exit_time
        daily_counts[trade_day] = daily_counts.get(trade_day, 0) + 1

        trades.append(
            {
                "entry_time": pd.Timestamp(times[entry_idx]),
                "exit_time": exit_time,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_reason": exit_reason,
                "entry_spread": spread,
                "pnl_points": pnl_points,
                "r_multiple": r_multiple,
                "return_pct": r_multiple * strategy.risk_per_trade * 100,
                "session_hour_utc": int(signal.hour_utc),
                "session_hour": int(signal.session_hour),
            }
        )

    return pd.DataFrame(trades)


def run_fast_bar_backtest_from_data(
    strategy: ResearchStrategy,
    start,
    end,
    frames: dict[str, pd.DataFrame],
    label: str = "",
    save_outputs: bool = True,
) -> dict:
    signals = generate_signals(frames, strategy)
    trades = simulate_m1_trades(frames["M1"], signals, strategy, start, end)
    summary = summarize_trades(trades, strategy, start=start, end=end)
    daily = daily_returns(trades, strategy)

    suffix = f"_{label}" if label else ""
    trades_path = RESULTS_DIR / f"{strategy.name}{suffix}_m1_trades.csv"
    summary_path = RESULTS_DIR / f"{strategy.name}{suffix}_m1_summary.json"
    daily_path = RESULTS_DIR / f"{strategy.name}{suffix}_m1_daily_returns.csv"
    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if not trades.empty:
            trades.to_csv(trades_path, index=False)
            daily.to_csv(daily_path, index=False)
        summary_path.write_text(json.dumps({**summary, "strategy_params": asdict(strategy)}, indent=2), encoding="utf-8")
    return {**summary, "trades_file": str(trades_path), "daily_file": str(daily_path), "summary_file": str(summary_path)}
