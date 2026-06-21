from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class ResearchStrategy:
    name: str
    symbol: str = "XAUUSD"
    style: str = "Hybrid"
    description: str = "XAUUSD hybrid multi-timeframe research strategy."
    entry_timeframe: str = "M5"
    confirmation_timeframe: str = "M15"
    trend_timeframe: str = "H1"
    risk_per_trade: float = 0.01
    fast_ma: int = 20
    slow_ma: int = 50
    trend_ma: int = 100
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    reward_risk: float = 2.0
    entry_mode: str = "ma_cross"
    breakout_lookback: int = 24
    pullback_lookback: int = 12
    rsi_period: int = 14
    use_rsi_filter: bool = False
    long_rsi_min: float = 50.0
    short_rsi_max: float = 50.0
    session_start_utc: int = 7
    session_end_utc: int = 21
    session_time_basis: str = "utc"
    server_time_offset_hours: int = 0
    max_trades_per_day: int = 3
    direction: str = "both"
    use_confirmation_filter: bool = True
    use_trend_filter: bool = True
    max_holding_hours: int = 48
    one_position_at_time: bool = True


def default_strategy() -> ResearchStrategy:
    return ResearchStrategy(
        name="xau_2025_hybrid_tick_v1",
        description="Hybrid XAUUSD strategy: M5 entry, M15 confirmation, H1 trend filter, tick-level fill simulation.",
        fast_ma=10,
        slow_ma=30,
        trend_ma=50,
        atr_stop_multiplier=1.5,
        reward_risk=1.8,
        entry_mode="ma_cross",
        breakout_lookback=24,
        pullback_lookback=12,
        rsi_period=14,
        use_rsi_filter=False,
        long_rsi_min=50.0,
        short_rsi_max=50.0,
        session_start_utc=0,
        session_end_utc=23,
        session_time_basis="utc",
        server_time_offset_hours=0,
        max_trades_per_day=5,
        direction="both",
        use_confirmation_filter=False,
        use_trend_filter=True,
        max_holding_hours=48,
        one_position_at_time=True,
    )


def load_strategy(path: Path) -> ResearchStrategy:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ResearchStrategy(**data)


def save_strategy(strategy: ResearchStrategy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(strategy), indent=2), encoding="utf-8")
