from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def rank_sessions(candles: pd.DataFrame, server_time_offset_hours: int = 0) -> pd.DataFrame:
    data = candles.copy()
    data["utc_time"] = data["time"] - pd.to_timedelta(server_time_offset_hours, unit="h")
    data["hour"] = data["utc_time"].dt.hour
    data["range"] = data["high"] - data["low"]
    return (
        data.groupby("hour")
        .agg(avg_range=("range", "mean"), avg_volume=("tick_volume", "mean"), bars=("time", "count"))
        .assign(score=lambda frame: frame["avg_range"] * frame["avg_volume"])
        .sort_values("score", ascending=False)
    )


def _entry_candles(candles_or_frames, strategy_spec: dict) -> pd.DataFrame:
    if isinstance(candles_or_frames, dict):
        timeframe = strategy_spec.get("entry_timeframe", "M5")
        return candles_or_frames[timeframe]
    return candles_or_frames


def run_backtest(candles_or_frames, strategy_spec: dict, server_time_offset_hours: int = 0) -> dict:
    # TODO: Replace this starter scoring with full signal generation and order-level simulation.
    candles = _entry_candles(candles_or_frames, strategy_spec)
    sessions = rank_sessions(candles, server_time_offset_hours=server_time_offset_hours)
    best_hour = int(sessions.index[0]) if not sessions.empty else None
    score = float(sessions.iloc[0]["score"]) if best_hour is not None else 0.0
    timeframe_counts = {}
    if isinstance(candles_or_frames, dict):
        timeframe_counts = {key: int(len(value)) for key, value in candles_or_frames.items()}
    return {
        "strategy": strategy_spec.get("name", "unknown"),
        "symbol": strategy_spec.get("symbol", "XAUUSD"),
        "bars": int(len(candles)),
        "source_timeframe": "M1",
        "entry_timeframe": strategy_spec.get("entry_timeframe", "M5"),
        "confirmation_timeframe": strategy_spec.get("confirmation_timeframe", "M15"),
        "trend_timeframe": strategy_spec.get("trend_timeframe", "H1"),
        "timeframe_bars": json.dumps(timeframe_counts),
        "server_time_offset_hours": int(server_time_offset_hours),
        "best_session_hour_utc": best_hour,
        "fitness": score,
    }


def save_results(result: dict, csv_path: Path, json_path: Path) -> None:
    pd.DataFrame([result]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
