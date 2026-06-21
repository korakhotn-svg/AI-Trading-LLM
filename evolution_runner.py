from __future__ import annotations

import argparse
import json
from datetime import datetime
from dataclasses import asdict

import optuna
import pandas as pd

from config import DUKASCOPY_SYMBOL, RESULTS_DIR, STRATEGIES_DIR, ensure_directories
from data_source import load_strategy_frames
from strategy_schema import ResearchStrategy
from tick_backtest_engine import run_tick_backtest_from_data
from tick_to_bars import load_ticks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python VPS runner: evolve strategy params on Dukascopy tick data.")
    parser.add_argument("--symbol", default=DUKASCOPY_SYMBOL)
    parser.add_argument("--backtest-start", default="2025-01-01")
    parser.add_argument("--backtest-end", default="2025-09-30")
    parser.add_argument("--forward-start", default="2025-10-01")
    parser.add_argument("--forward-end", default="2025-12-31")
    parser.add_argument("--trials", type=int, default=30)
    return parser.parse_args()


def candidate_from_trial(trial: optuna.Trial) -> ResearchStrategy:
    fast_ma = trial.suggest_int("fast_ma", 8, 40)
    slow_ma = trial.suggest_int("slow_ma", fast_ma + 5, 100)
    return ResearchStrategy(
        name=f"xau_tick_trial_{trial.number}",
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        trend_ma=trial.suggest_int("trend_ma", 80, 240),
        atr_period=trial.suggest_int("atr_period", 7, 28),
        atr_stop_multiplier=trial.suggest_float("atr_stop_multiplier", 1.0, 4.0),
        reward_risk=trial.suggest_float("reward_risk", 1.0, 4.0),
        entry_mode=trial.suggest_categorical("entry_mode", ["ma_cross", "breakout", "pullback"]),
        breakout_lookback=trial.suggest_int("breakout_lookback", 12, 96),
        pullback_lookback=trial.suggest_int("pullback_lookback", 6, 48),
        rsi_period=trial.suggest_int("rsi_period", 7, 28),
        use_rsi_filter=trial.suggest_categorical("use_rsi_filter", [False, True]),
        long_rsi_min=trial.suggest_float("long_rsi_min", 45, 65),
        short_rsi_max=trial.suggest_float("short_rsi_max", 35, 55),
        session_start_utc=trial.suggest_int("session_start_utc", 0, 12),
        session_end_utc=trial.suggest_int("session_end_utc", 13, 23),
        max_trades_per_day=trial.suggest_int("max_trades_per_day", 1, 5),
        direction=trial.suggest_categorical("direction", ["both", "long", "short"]),
        use_confirmation_filter=trial.suggest_categorical("use_confirmation_filter", [False, True]),
        use_trend_filter=trial.suggest_categorical("use_trend_filter", [True, False]),
        max_holding_hours=trial.suggest_int("max_holding_hours", 6, 96),
        one_position_at_time=True,
    )


def walk_forward_score(backtest: dict, forward: dict) -> float:
    if backtest.get("trades", 0) < 8 or forward.get("trades", 0) < 3:
        return -999.0
    forward_return = forward.get("total_return_pct", -100)
    forward_dd = abs(forward.get("max_drawdown_pct", 100))
    backtest_dd = abs(backtest.get("max_drawdown_pct", 100))
    overfit_gap = abs(backtest.get("expectancy_r", 0) - forward.get("expectancy_r", 0))
    trade_penalty = 0 if 10 <= forward.get("trades", 0) <= 80 else 5
    return float(forward_return - 1.5 * forward_dd - 0.5 * backtest_dd - 10 * overfit_gap - trade_penalty)


def main() -> None:
    args = parse_args()
    ensure_directories()
    backtest_start = datetime.strptime(args.backtest_start, "%Y-%m-%d").date()
    backtest_end = datetime.strptime(args.backtest_end, "%Y-%m-%d").date()
    forward_start = datetime.strptime(args.forward_start, "%Y-%m-%d").date()
    forward_end = datetime.strptime(args.forward_end, "%Y-%m-%d").date()
    results = []
    frames = load_strategy_frames(args.symbol, ["M5", "M15", "H1"])
    backtest_ticks = load_ticks(args.symbol, backtest_start, backtest_end)
    forward_ticks = load_ticks(args.symbol, forward_start, forward_end)

    def objective(trial: optuna.Trial) -> float:
        strategy = candidate_from_trial(trial)
        backtest = run_tick_backtest_from_data(
            strategy, backtest_start, backtest_end, frames, backtest_ticks, label=f"trial_{trial.number}_backtest"
        )
        forward = run_tick_backtest_from_data(
            strategy, forward_start, forward_end, frames, forward_ticks, label=f"trial_{trial.number}_forward"
        )
        score = walk_forward_score(backtest, forward)
        results.append(
            {
                "trial": trial.number,
                "walk_forward_score": score,
                "backtest_trades": backtest.get("trades", 0),
                "backtest_return_pct": backtest.get("total_return_pct"),
                "backtest_max_dd_pct": backtest.get("max_drawdown_pct"),
                "backtest_profit_factor": backtest.get("profit_factor"),
                "forward_trades": forward.get("trades", 0),
                "forward_return_pct": forward.get("total_return_pct"),
                "forward_apr_pct": forward.get("apr_pct"),
                "forward_max_dd_pct": forward.get("max_drawdown_pct"),
                "forward_profit_factor": forward.get("profit_factor"),
                "forward_avg_spread": forward.get("avg_spread"),
                **trial.params,
            }
        )
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    output = RESULTS_DIR / "evolution_results.csv"
    ranked = pd.DataFrame(results).sort_values("walk_forward_score", ascending=False)
    ranked.to_csv(output, index=False)
    best_strategy = candidate_from_trial(study.best_trial)
    best_path = STRATEGIES_DIR / "best_evolved_strategy.json"
    best_path.write_text(json.dumps(asdict(best_strategy), indent=2), encoding="utf-8")
    print(f"best_value={study.best_value}")
    print(f"best_params={study.best_params}")
    print(f"results={output}")
    print(f"best_strategy={best_path}")


if __name__ == "__main__":
    main()
