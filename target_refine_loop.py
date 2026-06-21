from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd

from config import DUKASCOPY_SYMBOL, RESULTS_DIR, STRATEGIES_DIR, ensure_directories
from data_source import load_strategy_frames
from fast_bar_backtest_engine import run_fast_bar_backtest_from_data
from strategy_schema import ResearchStrategy, load_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token-efficient loop toward target daily profit and max DD.")
    parser.add_argument("--symbol", default=DUKASCOPY_SYMBOL)
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--forward-start", default="2026-01-01")
    parser.add_argument("--forward-end", default="2026-06-19")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--trials-per-cycle", type=int, default=8)
    parser.add_argument("--target-min-daily-pct", type=float, default=1.0)
    parser.add_argument("--target-max-daily-pct", type=float, default=2.0)
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--engine", choices=["m1"], default="m1", help="Optimization engine. m1 is fast and spread-aware.")
    parser.add_argument(
        "--seed-strategies",
        default="strategies/xau_llm_strategy_v2.json,strategies/best_evolved_strategy.json",
        help="Comma-separated strategy JSON files to evaluate before random search.",
    )
    return parser.parse_args()


def as_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def suggest_near_int(trial: optuna.Trial, name: str, base: int, low: int, high: int, span: int) -> int:
    delta = trial.suggest_int(name + "_delta", -span, span)
    return max(low, min(high, int(base + delta)))


def suggest_near_float(
    trial: optuna.Trial, name: str, base: float, low: float, high: float, span: float
) -> float:
    delta = trial.suggest_float(name + "_delta", -span, span)
    return max(low, min(high, float(base + delta)))


def unique_choices(values: list) -> list:
    choices = []
    for value in values:
        if value not in choices:
            choices.append(value)
    return choices


def candidate_from_trial(trial: optuna.Trial, cycle: int, seed_pool: list[ResearchStrategy]) -> ResearchStrategy:
    profile = trial.suggest_categorical("profile", ["seed_mutation", "aggressive_breakout", "session_probe"])
    base = seed_pool[trial.suggest_int("seed_index", 0, len(seed_pool) - 1)] if seed_pool else ResearchStrategy(name="base")
    prefix = f"{profile}_"

    if profile == "aggressive_breakout":
        entry_mode = "breakout"
        fast_ma = trial.suggest_int(prefix + "fast_ma", 8, 28)
        slow_ma = max(trial.suggest_int(prefix + "slow_ma", 30, 75), fast_ma + 8)
        trend_ma = trial.suggest_int(prefix + "trend_ma", 55, 125)
        atr_stop_multiplier = trial.suggest_float(prefix + "atr_stop_multiplier", 1.0, 2.6)
        reward_risk = trial.suggest_float(prefix + "reward_risk", 1.1, 2.4)
        breakout_lookback = trial.suggest_int(prefix + "breakout_lookback", 5, 22)
        max_holding_hours = trial.suggest_int(prefix + "max_holding_hours", 4, 28)
        max_trades_per_day = trial.suggest_int(prefix + "max_trades_per_day", 4, 8)
        use_confirmation_filter = trial.suggest_categorical(prefix + "use_confirmation_filter", [False, True])
        use_trend_filter = trial.suggest_categorical(prefix + "use_trend_filter", [True, False])
    else:
        entry_mode = trial.suggest_categorical(prefix + "entry_mode", ["breakout", "pullback", "ma_cross"])
        fast_ma = suggest_near_int(trial, prefix + "fast_ma", base.fast_ma, 6, 45, 12)
        slow_ma = suggest_near_int(trial, prefix + "slow_ma", max(base.slow_ma, fast_ma + 5), fast_ma + 5, 100, 18)
        trend_ma = suggest_near_int(trial, prefix + "trend_ma", base.trend_ma, 45, 150, 25)
        atr_stop_multiplier = suggest_near_float(
            trial, prefix + "atr_stop_multiplier", base.atr_stop_multiplier, 0.9, 3.8, 0.8
        )
        reward_risk = suggest_near_float(trial, prefix + "reward_risk", base.reward_risk, 1.0, 3.4, 0.7)
        breakout_lookback = suggest_near_int(trial, prefix + "breakout_lookback", base.breakout_lookback, 4, 35, 9)
        max_holding_hours = suggest_near_int(trial, prefix + "max_holding_hours", base.max_holding_hours, 4, 56, 16)
        max_trades_per_day = suggest_near_int(trial, prefix + "max_trades_per_day", base.max_trades_per_day, 2, 8, 3)
        use_confirmation_filter = trial.suggest_categorical(prefix + "use_confirmation_filter", [False, True])
        use_trend_filter = trial.suggest_categorical(prefix + "use_trend_filter", [False, True])

    if profile == "session_probe":
        session_start_utc = trial.suggest_int(prefix + "session_start_utc", 0, 8)
        session_end_utc = trial.suggest_int(prefix + "session_end_utc", 12, 23)
    else:
        session_start_utc = suggest_near_int(trial, prefix + "session_start_utc", base.session_start_utc, 0, 8, 3)
        session_end_utc = suggest_near_int(trial, prefix + "session_end_utc", base.session_end_utc, 12, 23, 3)

    return ResearchStrategy(
        name=f"target_loop_c{cycle}_t{trial.number}",
        style="Hybrid-LLM-Loop",
        description="Token-efficient local search candidate for XAUUSD daily profit/DD target.",
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        trend_ma=trend_ma,
        atr_period=suggest_near_int(trial, prefix + "atr_period", base.atr_period, 5, 21, 6),
        atr_stop_multiplier=atr_stop_multiplier,
        reward_risk=reward_risk,
        entry_mode=entry_mode,
        breakout_lookback=breakout_lookback,
        pullback_lookback=suggest_near_int(trial, prefix + "pullback_lookback", base.pullback_lookback, 5, 35, 8),
        rsi_period=suggest_near_int(trial, prefix + "rsi_period", base.rsi_period, 7, 30, 6),
        use_rsi_filter=trial.suggest_categorical(prefix + "use_rsi_filter", [False, True]),
        long_rsi_min=suggest_near_float(trial, prefix + "long_rsi_min", base.long_rsi_min, 48, 64, 7),
        short_rsi_max=suggest_near_float(trial, prefix + "short_rsi_max", base.short_rsi_max, 36, 52, 7),
        session_start_utc=session_start_utc,
        session_end_utc=session_end_utc,
        max_trades_per_day=max_trades_per_day,
        direction=trial.suggest_categorical(prefix + "direction", ["both", "long", "short"]),
        use_confirmation_filter=use_confirmation_filter,
        use_trend_filter=use_trend_filter,
        max_holding_hours=max_holding_hours,
        one_position_at_time=True,
    )


def add_target_metrics(result: dict, start, end, prefix: str) -> dict:
    days = max((end - start).days + 1, 1)
    total = float(result.get("total_return_pct") or 0)
    trades = int(result.get("trades") or 0)
    return {
        f"{prefix}_trades": trades,
        f"{prefix}_return_pct": total,
        f"{prefix}_calendar_daily_pct": total / days,
        f"{prefix}_active_trade_daily_pct": total / max(trades, 1),
        f"{prefix}_trade_day_daily_pct": float(result.get("avg_trade_day_return_pct") or 0),
        f"{prefix}_trading_days": int(result.get("trading_days") or 0),
        f"{prefix}_positive_day_rate": float(result.get("positive_day_rate") or 0),
        f"{prefix}_max_dd_pct": float(result.get("max_drawdown_pct") or 0),
        f"{prefix}_profit_factor": result.get("profit_factor"),
        f"{prefix}_avg_spread": result.get("avg_spread"),
    }


def target_score(row: dict, target_min: float, target_max: float, max_dd: float) -> float:
    daily = row["forward_trade_day_daily_pct"]
    dd = abs(row["forward_max_dd_pct"])
    forward_return = row["forward_return_pct"]
    train_return = row["train_return_pct"]
    trades = row["forward_trades"]
    positive_day_rate = row["forward_positive_day_rate"]
    target_center = (target_min + target_max) / 2
    target_penalty = abs(daily - target_center) * 100
    if target_min <= daily <= target_max:
        target_penalty *= 0.25
        target_bonus = 120.0
    else:
        target_bonus = 0.0
    dd_penalty = max(0.0, dd - max_dd) * 5
    low_trade_penalty = max(0, 12 - trades) * 2
    overfit_penalty = abs(train_return - forward_return) * 0.2
    consistency_bonus = positive_day_rate * 3
    return float(
        min(forward_return, 250) * 0.45
        - dd * 1.5
        - target_penalty
        - dd_penalty
        - low_trade_penalty
        - overfit_penalty
        + consistency_bonus
        + target_bonus
    )


def main() -> None:
    args = parse_args()
    ensure_directories()
    train_start, train_end = as_date(args.train_start), as_date(args.train_end)
    forward_start, forward_end = as_date(args.forward_start), as_date(args.forward_end)

    print("Loading cached bars once...")
    frames = load_strategy_frames(args.symbol, ["M1", "M5", "M15", "H1"])

    rows = []
    best_score = -1e9
    best_strategy = None
    seed_pool: list[ResearchStrategy] = []

    def evaluate_strategy(strategy: ResearchStrategy, cycle: int, trial_label) -> float:
        nonlocal best_score, best_strategy
        train = run_fast_bar_backtest_from_data(strategy, train_start, train_end, frames, save_outputs=False)
        forward = run_fast_bar_backtest_from_data(strategy, forward_start, forward_end, frames, save_outputs=False)
        row = {
            "cycle": cycle,
            "trial": trial_label,
            "strategy_name": strategy.name,
            **add_target_metrics(train, train_start, train_end, "train"),
            **add_target_metrics(forward, forward_start, forward_end, "forward"),
            **asdict(strategy),
        }
        row["target_score"] = target_score(
            row, args.target_min_daily_pct, args.target_max_daily_pct, args.max_dd_pct
        )
        row["target_hit"] = (
            args.target_min_daily_pct <= row["forward_trade_day_daily_pct"] <= args.target_max_daily_pct
            and abs(row["forward_max_dd_pct"]) <= args.max_dd_pct
            and row["forward_return_pct"] > 0
        )
        rows.append(row)
        if row["target_score"] > best_score:
            best_score = row["target_score"]
            best_strategy = strategy
            print(
                f"New best score={best_score:.2f} strategy={strategy.name} "
                f"forward_return={row['forward_return_pct']:.2f}% "
                f"trade_day_daily={row['forward_trade_day_daily_pct']:.2f}% "
                f"dd={row['forward_max_dd_pct']:.2f}% trades={row['forward_trades']}"
            )
        return row["target_score"]

    print("Evaluating seed strategies...")
    for raw_path in [item.strip() for item in args.seed_strategies.split(",") if item.strip()]:
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            seed_strategy = load_strategy(path)
            seed_pool.append(seed_strategy)
            evaluate_strategy(seed_strategy, 0, f"seed:{path.name}")
    if best_strategy and all(item.name != best_strategy.name for item in seed_pool):
        seed_pool.append(best_strategy)
    if rows:
        output = RESULTS_DIR / "target_refine_loop_results.csv"
        pd.DataFrame(rows).sort_values("target_score", ascending=False).to_csv(output, index=False)
        if best_strategy:
            best_path = STRATEGIES_DIR / "target_loop_best_strategy.json"
            best_path.write_text(json.dumps(asdict(best_strategy), indent=2), encoding="utf-8")

    for cycle in range(1, args.cycles + 1):
        print(f"Cycle {cycle}/{args.cycles}")

        def objective(trial: optuna.Trial) -> float:
            strategy = candidate_from_trial(trial, cycle, seed_pool)
            return evaluate_strategy(strategy, cycle, trial.number)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.trials_per_cycle)

        output = RESULTS_DIR / "target_refine_loop_results.csv"
        pd.DataFrame(rows).sort_values("target_score", ascending=False).to_csv(output, index=False)
        if best_strategy:
            if all(item.name != best_strategy.name for item in seed_pool):
                seed_pool.append(best_strategy)
            best_path = STRATEGIES_DIR / "target_loop_best_strategy.json"
            best_path.write_text(json.dumps(asdict(best_strategy), indent=2), encoding="utf-8")
            print(f"checkpoint={output}")
            print(f"best_strategy={best_path}")

        if rows and pd.DataFrame(rows).sort_values("target_score", ascending=False).iloc[0]["target_hit"]:
            print("Target hit. Stopping loop.")
            break

    print(pd.DataFrame(rows).sort_values("target_score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
