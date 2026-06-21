from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import DUKASCOPY_SYMBOL, RESULTS_DIR
from data_source import load_strategy_frames
from strategy_schema import ResearchStrategy
from tick_to_bars import load_ticks


TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
}


def add_indicators(frames: dict[str, pd.DataFrame], strategy: ResearchStrategy) -> dict[str, pd.DataFrame]:
    prepared = {key: value.copy().sort_values("time") for key, value in frames.items()}
    entry = prepared[strategy.entry_timeframe]
    entry["fast_ma"] = entry["close"].rolling(strategy.fast_ma).mean()
    entry["slow_ma"] = entry["close"].rolling(strategy.slow_ma).mean()
    entry["prev_fast_ma"] = entry["fast_ma"].shift(1)
    entry["prev_slow_ma"] = entry["slow_ma"].shift(1)
    entry["tr"] = (entry["high"] - entry["low"]).abs()
    entry["atr"] = entry["tr"].rolling(strategy.atr_period).mean()
    entry["breakout_high"] = entry["high"].shift(1).rolling(strategy.breakout_lookback).max()
    entry["breakout_low"] = entry["low"].shift(1).rolling(strategy.breakout_lookback).min()
    entry["pullback_high"] = entry["high"].shift(1).rolling(strategy.pullback_lookback).max()
    entry["pullback_low"] = entry["low"].shift(1).rolling(strategy.pullback_lookback).min()
    delta = entry["close"].diff()
    gain = delta.clip(lower=0).rolling(strategy.rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(strategy.rsi_period).mean()
    rs = gain / loss.replace(0, pd.NA)
    entry["rsi"] = 100 - (100 / (1 + rs))
    prepared[strategy.entry_timeframe] = entry

    confirmation = prepared[strategy.confirmation_timeframe]
    confirmation["confirm_ma"] = confirmation["close"].rolling(strategy.slow_ma).mean()
    prepared[strategy.confirmation_timeframe] = confirmation

    trend = prepared[strategy.trend_timeframe]
    trend["trend_ma"] = trend["close"].rolling(strategy.trend_ma).mean()
    prepared[strategy.trend_timeframe] = trend
    return prepared


def align_higher_timeframes(entry: pd.DataFrame, confirmation: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        entry.sort_values("time"),
        confirmation[["time", "close", "confirm_ma"]].rename(
            columns={"close": "confirmation_close", "confirm_ma": "confirmation_ma"}
        ).sort_values("time"),
        on="time",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("time"),
        trend[["time", "close", "trend_ma"]].rename(columns={"close": "trend_close"}).sort_values("time"),
        on="time",
        direction="backward",
    )
    return merged


def generate_signals(frames: dict[str, pd.DataFrame], strategy: ResearchStrategy) -> pd.DataFrame:
    prepared = add_indicators(frames, strategy)
    entry = align_higher_timeframes(
        prepared[strategy.entry_timeframe],
        prepared[strategy.confirmation_timeframe],
        prepared[strategy.trend_timeframe],
    ).dropna()
    entry["hour_utc"] = entry["time"].dt.hour
    if strategy.session_time_basis == "broker_server":
        adjusted_time = entry["time"] + pd.to_timedelta(strategy.server_time_offset_hours, unit="h")
        entry["session_hour"] = adjusted_time.dt.hour
    else:
        entry["session_hour"] = entry["hour_utc"]
    in_session = entry["session_hour"].between(strategy.session_start_utc, strategy.session_end_utc - 1)
    cross_up = (entry["prev_fast_ma"] <= entry["prev_slow_ma"]) & (entry["fast_ma"] > entry["slow_ma"])
    cross_down = (entry["prev_fast_ma"] >= entry["prev_slow_ma"]) & (entry["fast_ma"] < entry["slow_ma"])
    breakout_up = entry["close"] > entry["breakout_high"]
    breakout_down = entry["close"] < entry["breakout_low"]
    pullback_up = (entry["fast_ma"] > entry["slow_ma"]) & (entry["low"] <= entry["pullback_low"]) & (entry["close"] > entry["fast_ma"])
    pullback_down = (entry["fast_ma"] < entry["slow_ma"]) & (entry["high"] >= entry["pullback_high"]) & (entry["close"] < entry["fast_ma"])
    if strategy.entry_mode == "breakout":
        long_entry = breakout_up
        short_entry = breakout_down
    elif strategy.entry_mode == "pullback":
        long_entry = pullback_up
        short_entry = pullback_down
    else:
        long_entry = cross_up
        short_entry = cross_down
    long_filter = pd.Series(True, index=entry.index)
    short_filter = pd.Series(True, index=entry.index)
    if strategy.use_confirmation_filter:
        long_filter &= entry["confirmation_close"] > entry["confirmation_ma"]
        short_filter &= entry["confirmation_close"] < entry["confirmation_ma"]
    if strategy.use_trend_filter:
        long_filter &= entry["trend_close"] > entry["trend_ma"]
        short_filter &= entry["trend_close"] < entry["trend_ma"]
    if strategy.use_rsi_filter:
        long_filter &= entry["rsi"] >= strategy.long_rsi_min
        short_filter &= entry["rsi"] <= strategy.short_rsi_max
    entry["signal"] = 0
    if strategy.direction in {"both", "long"}:
        entry.loc[in_session & long_entry & long_filter, "signal"] = 1
    if strategy.direction in {"both", "short"}:
        entry.loc[in_session & short_entry & short_filter, "signal"] = -1
    signals = entry[entry["signal"] != 0].copy()
    minutes = TIMEFRAME_MINUTES.get(strategy.entry_timeframe, 5)
    signals["execution_time"] = signals["time"] + pd.to_timedelta(minutes, unit="min")
    return signals


def simulate_tick_trades(ticks: pd.DataFrame, signals: pd.DataFrame, strategy: ResearchStrategy) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    tick_time_values = pd.to_datetime(ticks["time"], utc=True, format="mixed").dt.tz_convert(None).to_numpy()
    sort_order = np.argsort(tick_time_values)
    tick_times = tick_time_values[sort_order]
    bids = ticks["bid"].to_numpy(dtype=float)[sort_order]
    asks = ticks["ask"].to_numpy(dtype=float)[sort_order]
    spreads = ticks["spread"].to_numpy(dtype=float)[sort_order]
    trades = []
    trades_today = {}
    next_allowed_entry_time = None

    for signal in signals.itertuples(index=False):
        signal_time = getattr(signal, "execution_time", signal.time)
        if strategy.one_position_at_time and next_allowed_entry_time is not None and signal_time < next_allowed_entry_time:
            continue
        day = signal_time.date().isoformat()
        trades_today[day] = trades_today.get(day, 0)
        if trades_today[day] >= strategy.max_trades_per_day:
            continue

        start_idx = int(np.searchsorted(tick_times, np.datetime64(signal_time), side="left"))
        if start_idx >= len(tick_times):
            continue

        direction = int(signal.signal)
        entry_price = float(asks[start_idx] if direction == 1 else bids[start_idx])
        entry_spread = float(spreads[start_idx])
        atr = max(float(signal.atr), 0.01)
        stop_distance = atr * strategy.atr_stop_multiplier
        target_distance = stop_distance * strategy.reward_risk
        stop = entry_price - direction * stop_distance
        target = entry_price + direction * target_distance
        max_exit_time = signal_time + pd.to_timedelta(strategy.max_holding_hours, unit="h")
        exit_price = None
        exit_time = None
        outcome = "open"

        # Tick-level fill simulation after bar signal.
        for idx in range(start_idx, len(tick_times)):
            bid = bids[idx]
            ask = asks[idx]
            current_time = pd.Timestamp(tick_times[idx]).to_pydatetime()
            if current_time >= max_exit_time:
                exit_price = float(bid if direction == 1 else ask)
                exit_time = tick_times[idx]
                outcome = "time_exit"
                break
            if direction == 1:
                if bid <= stop:
                    exit_price, exit_time, outcome = stop, tick_times[idx], "stop"
                    break
                if bid >= target:
                    exit_price, exit_time, outcome = target, tick_times[idx], "target"
                    break
            else:
                if ask >= stop:
                    exit_price, exit_time, outcome = stop, tick_times[idx], "stop"
                    break
                if ask <= target:
                    exit_price, exit_time, outcome = target, tick_times[idx], "target"
                    break

        if exit_price is None:
            exit_price = float(bids[-1] if direction == 1 else asks[-1])
            exit_time = tick_times[-1]
            outcome = "end_of_data"

        pnl_points = (exit_price - entry_price) * direction
        risk_points = stop_distance
        r_multiple = pnl_points / risk_points if risk_points else 0
        trades_today[day] += 1
        next_allowed_entry_time = pd.Timestamp(exit_time).to_pydatetime()
        trades.append(
            {
                "strategy": strategy.name,
                "signal_time": signal.time,
                "entry_time": signal_time,
                "exit_time": exit_time,
                "direction": "long" if direction == 1 else "short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop": stop,
                "target": target,
                "outcome": outcome,
                "entry_spread": entry_spread,
                "pnl_points": pnl_points,
                "r_multiple": r_multiple,
                "return_pct": r_multiple * strategy.risk_per_trade * 100,
                "session_hour_utc": int(signal.hour_utc),
                "session_hour": int(signal.session_hour),
            }
        )

    return pd.DataFrame(trades)


def daily_returns(trades: pd.DataFrame, strategy: ResearchStrategy) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "trades", "daily_return_pct", "equity_pct", "drawdown_pct"])
    data = trades.copy()
    data["date"] = pd.to_datetime(data["exit_time"]).dt.date.astype(str)
    daily = (
        data.groupby("date")
        .agg(
            trades=("r_multiple", "count"),
            daily_return_pct=("return_pct", "sum"),
            avg_spread=("entry_spread", "mean"),
            net_r=("r_multiple", "sum"),
        )
        .reset_index()
    )
    daily["equity_pct"] = daily["daily_return_pct"].cumsum()
    daily["drawdown_pct"] = daily["equity_pct"] - daily["equity_pct"].cummax()
    return daily


def summarize_trades(trades: pd.DataFrame, strategy: ResearchStrategy, start=None, end=None) -> dict:
    if trades.empty:
        return {
            "strategy": strategy.name,
            "trades": 0,
            "fitness": -999,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trading_days": 0,
            "avg_trade_day_return_pct": 0.0,
            "best_daily_return_pct": 0.0,
            "worst_daily_return_pct": 0.0,
            "positive_day_rate": 0.0,
            "note": "No trades generated.",
        }

    equity = trades["r_multiple"].cumsum()
    drawdown = equity - equity.cummax()
    daily = daily_returns(trades, strategy)
    total_return_pct = float(trades["return_pct"].sum())
    max_drawdown_pct = float(daily["drawdown_pct"].min()) if not daily.empty else 0.0
    if start and end:
        days = max((end - start).days + 1, 1)
    else:
        days = max((pd.to_datetime(trades["exit_time"]).max() - pd.to_datetime(trades["entry_time"]).min()).days, 1)
    apr_pct = ((1 + total_return_pct / 100) ** (365 / days) - 1) * 100 if total_return_pct > -100 else -100.0
    gross_profit = trades.loc[trades["r_multiple"] > 0, "r_multiple"].sum()
    gross_loss = abs(trades.loc[trades["r_multiple"] < 0, "r_multiple"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    win_rate = float((trades["r_multiple"] > 0).mean())
    expectancy = float(trades["r_multiple"].mean())
    max_drawdown = float(drawdown.min())
    trading_days = int(len(daily))
    avg_trade_day_return_pct = float(daily["daily_return_pct"].mean()) if trading_days else 0.0
    best_daily_return_pct = float(daily["daily_return_pct"].max()) if trading_days else 0.0
    worst_daily_return_pct = float(daily["daily_return_pct"].min()) if trading_days else 0.0
    positive_day_rate = float((daily["daily_return_pct"] > 0).mean()) if trading_days else 0.0
    fitness = expectancy * len(trades) + max_drawdown
    return {
        "strategy": strategy.name,
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy,
        "net_r": float(trades["r_multiple"].sum()),
        "total_return_pct": total_return_pct,
        "apr_pct": float(apr_pct),
        "max_drawdown_r": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "trading_days": trading_days,
        "avg_trade_day_return_pct": avg_trade_day_return_pct,
        "best_daily_return_pct": best_daily_return_pct,
        "worst_daily_return_pct": worst_daily_return_pct,
        "positive_day_rate": positive_day_rate,
        "avg_spread": float(trades["entry_spread"].mean()),
        "best_session_hour_utc": int(trades.groupby("session_hour_utc")["r_multiple"].sum().idxmax()),
        "best_session_hour": int(trades.groupby("session_hour")["r_multiple"].sum().idxmax()),
        "fitness": float(fitness),
    }


def run_tick_backtest_from_data(
    strategy: ResearchStrategy,
    start,
    end,
    frames: dict[str, pd.DataFrame],
    ticks: pd.DataFrame,
    label: str = "",
    save_outputs: bool = True,
) -> dict:
    signals = generate_signals(frames, strategy)
    signals = signals[(signals["time"].dt.date >= start) & (signals["time"].dt.date <= end)]
    trades = simulate_tick_trades(ticks, signals, strategy)
    summary = summarize_trades(trades, strategy, start=start, end=end)
    daily = daily_returns(trades, strategy)

    suffix = f"_{label}" if label else ""
    trades_path = RESULTS_DIR / f"{strategy.name}{suffix}_trades.csv"
    summary_path = RESULTS_DIR / f"{strategy.name}{suffix}_summary.json"
    daily_path = RESULTS_DIR / f"{strategy.name}{suffix}_daily_returns.csv"
    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if not trades.empty:
            trades.to_csv(trades_path, index=False)
            daily.to_csv(daily_path, index=False)
        summary_path.write_text(json.dumps({**summary, "strategy_params": asdict(strategy)}, indent=2), encoding="utf-8")
    return {**summary, "trades_file": str(trades_path), "daily_file": str(daily_path), "summary_file": str(summary_path)}


def run_tick_backtest(strategy: ResearchStrategy, start, end, symbol: str = DUKASCOPY_SYMBOL, label: str = "") -> dict:
    frames = load_strategy_frames(
        symbol,
        [strategy.entry_timeframe, strategy.confirmation_timeframe, strategy.trend_timeframe],
    )
    ticks = load_ticks(symbol, start, end)
    return run_tick_backtest_from_data(strategy, start, end, frames, ticks, label=label)
