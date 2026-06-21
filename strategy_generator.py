from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from config import CONFIRMATION_TIMEFRAME, ENTRY_TIMEFRAME, RISK_PER_TRADE, SYMBOL, TREND_TIMEFRAME


@dataclass
class StrategySpec:
    name: str
    symbol: str
    style: str
    entry_timeframe: str
    confirmation_timeframe: str
    trend_timeframe: str
    entry_logic: str
    confirmation_logic: str
    trend_filter: str
    exit_logic: str
    risk_per_trade: float
    session_selection: str = "auto"


def fallback_strategy() -> StrategySpec:
    return StrategySpec(
        name="xau_hybrid_session_breakout_v1",
        symbol=SYMBOL,
        style="Hybrid",
        entry_timeframe=ENTRY_TIMEFRAME,
        confirmation_timeframe=CONFIRMATION_TIMEFRAME,
        trend_timeframe=TREND_TIMEFRAME,
        entry_logic="Enter on M5 pullback continuation after volatility expansion.",
        confirmation_logic="Require M15 momentum and market structure confirmation.",
        trend_filter="Trade only with the H1 trend bias.",
        exit_logic="Use ATR stop, partial take profit, and trailing stop after first target.",
        risk_per_trade=RISK_PER_TRADE,
    )


def generate_strategy(prompt_context: str = "") -> StrategySpec:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # TODO: Enable live AI generation after OPENAI_API_KEY is added to .env.
        return fallback_strategy()

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=(
                "Return one JSON object for an XAUUSD hybrid strategy. "
                f"Use {ENTRY_TIMEFRAME} entry, {CONFIRMATION_TIMEFRAME} confirmation, "
                f"{TREND_TIMEFRAME} trend filter, risk {RISK_PER_TRADE:.0%}, and automatic session choice. "
                f"Context: {prompt_context}"
            ),
        )
        return StrategySpec(**json.loads(response.output_text))
    except Exception as exc:
        # Keep research runs alive when the API key has no quota, billing is inactive,
        # or the model response is temporarily unavailable.
        print(f"OpenAI strategy generation unavailable, using fallback strategy: {exc}")
        return fallback_strategy()


def save_strategy(strategy: StrategySpec, path: Path) -> None:
    path.write_text(json.dumps(asdict(strategy), indent=2), encoding="utf-8")
