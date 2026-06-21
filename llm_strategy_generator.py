from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from openai import OpenAI

from config import RESULTS_DIR
from strategy_schema import ResearchStrategy, default_strategy


ALLOWED_ENTRY_MODES = {"ma_cross", "breakout", "pullback"}
ALLOWED_DIRECTIONS = {"both", "long", "short"}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_strategy(data: dict, name: str = "xau_llm_strategy_v1") -> ResearchStrategy:
    base = asdict(default_strategy())
    base.update(data)
    base["name"] = str(base.get("name") or name)
    base["symbol"] = "XAUUSD"
    base["style"] = "Hybrid-LLM"
    base["entry_timeframe"] = "M5"
    base["confirmation_timeframe"] = "M15"
    base["trend_timeframe"] = "H1"
    base["risk_per_trade"] = 0.01
    base["entry_mode"] = base.get("entry_mode") if base.get("entry_mode") in ALLOWED_ENTRY_MODES else "breakout"
    base["direction"] = base.get("direction") if base.get("direction") in ALLOWED_DIRECTIONS else "both"
    base["fast_ma"] = int(clamp(int(base.get("fast_ma", 20)), 5, 80))
    base["slow_ma"] = int(clamp(int(base.get("slow_ma", 50)), base["fast_ma"] + 3, 180))
    base["trend_ma"] = int(clamp(int(base.get("trend_ma", 100)), 20, 300))
    base["atr_period"] = int(clamp(int(base.get("atr_period", 14)), 5, 50))
    base["atr_stop_multiplier"] = float(clamp(float(base.get("atr_stop_multiplier", 2.0)), 0.8, 6.0))
    base["reward_risk"] = float(clamp(float(base.get("reward_risk", 2.0)), 0.8, 6.0))
    base["breakout_lookback"] = int(clamp(int(base.get("breakout_lookback", 24)), 6, 160))
    base["pullback_lookback"] = int(clamp(int(base.get("pullback_lookback", 12)), 4, 80))
    base["rsi_period"] = int(clamp(int(base.get("rsi_period", 14)), 5, 50))
    base["long_rsi_min"] = float(clamp(float(base.get("long_rsi_min", 50)), 40, 75))
    base["short_rsi_max"] = float(clamp(float(base.get("short_rsi_max", 50)), 25, 60))
    base["session_start_utc"] = int(clamp(int(base.get("session_start_utc", 0)), 0, 22))
    base["session_end_utc"] = int(clamp(int(base.get("session_end_utc", 23)), base["session_start_utc"] + 1, 23))
    base["max_trades_per_day"] = int(clamp(int(base.get("max_trades_per_day", 3)), 1, 8))
    base["max_holding_hours"] = int(clamp(int(base.get("max_holding_hours", 48)), 1, 120))
    base["one_position_at_time"] = bool(base.get("one_position_at_time", True))
    base["use_confirmation_filter"] = bool(base.get("use_confirmation_filter", False))
    base["use_trend_filter"] = bool(base.get("use_trend_filter", True))
    base["use_rsi_filter"] = bool(base.get("use_rsi_filter", False))
    base["description"] = str(
        base.get("description")
        or "LLM-designed XAUUSD hybrid strategy using M5 entry, M15 confirmation, H1 trend filter, tick fills."
    )
    return ResearchStrategy(**base)


def latest_performance_context() -> str:
    report = RESULTS_DIR / "xau_tick_trial_13_walk_forward_report.csv"
    evolution = RESULTS_DIR / "evolution_results.csv"
    chunks = []
    if report.exists():
        chunks.append("Latest walk-forward report:\n" + pd.read_csv(report).to_string(index=False))
    if evolution.exists():
        top = pd.read_csv(evolution).head(8)
        chunks.append("Top evolution rows:\n" + top.to_string(index=False))
    return "\n\n".join(chunks) or "No performance report available yet."


def offline_llm_style_strategy(context: str) -> ResearchStrategy:
    # Fallback when API quota is unavailable. It intentionally reacts to known weak points:
    # low forward PF, low trade count, and too much dependence on long holds.
    return ResearchStrategy(
        name="xau_llm_breakout_research_v1",
        style="Hybrid-LLM",
        description=(
            "LLM-style fallback: volatility breakout strategy for XAUUSD. "
            "M5 breakout entries, optional RSI momentum, H1 trend alignment, shorter max holding."
        ),
        entry_mode="breakout",
        fast_ma=24,
        slow_ma=60,
        trend_ma=90,
        atr_period=10,
        atr_stop_multiplier=2.4,
        reward_risk=2.2,
        breakout_lookback=18,
        pullback_lookback=16,
        rsi_period=14,
        use_rsi_filter=True,
        long_rsi_min=55,
        short_rsi_max=45,
        session_start_utc=0,
        session_end_utc=21,
        max_trades_per_day=3,
        direction="both",
        use_confirmation_filter=False,
        use_trend_filter=True,
        max_holding_hours=24,
        one_position_at_time=True,
    )


def generate_llm_strategy(context: str, model: str = "gpt-5-mini") -> ResearchStrategy:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return offline_llm_style_strategy(context)

    client = OpenAI(api_key=api_key)
    prompt = f"""
Design one XAUUSD Hybrid-LLM strategy as strict JSON only.

Environment:
- Data: Dukascopy XAUUSD 2025 ticks.
- Signal bars: M5 entry, M15 confirmation, H1 trend.
- Execution: tick-level bid/ask fill, spread-aware, one-position-at-time.
- Risk: 1% per trade.
- Goal: robust walk-forward, not backtest-only overfit.

Allowed entry_mode: ma_cross, breakout, pullback.
Allowed direction: both, long, short.
Prefer enough trades, controlled drawdown, realistic max_holding_hours.

Recent performance:
{context}

Return JSON fields matching ResearchStrategy:
name, description, entry_mode, fast_ma, slow_ma, trend_ma, atr_period,
atr_stop_multiplier, reward_risk, breakout_lookback, pullback_lookback,
rsi_period, use_rsi_filter, long_rsi_min, short_rsi_max,
session_start_utc, session_end_utc, max_trades_per_day, direction,
use_confirmation_filter, use_trend_filter, max_holding_hours, one_position_at_time.
"""
    try:
        response = client.responses.create(model=model, input=prompt)
        text = response.output_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.removeprefix("json").strip()
        return normalize_strategy(json.loads(text), name="xau_llm_strategy_v1")
    except Exception as exc:
        print(f"LLM strategy generation unavailable, using offline LLM-style strategy: {exc}")
        return offline_llm_style_strategy(context)


def save_llm_strategy(strategy: ResearchStrategy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(strategy), indent=2), encoding="utf-8")
