from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, STRATEGIES_DIR, ensure_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create monthly/quarterly robustness report from tick-confirm trades.")
    parser.add_argument("--trades", default="results/target_loop_c1_t20_tick_confirm_forward_2026_trades.csv")
    parser.add_argument("--strategy", default="strategies/target_loop_best_strategy.json")
    parser.add_argument("--label", default="target_loop_c1_t20_robustness_2026")
    parser.add_argument("--max-dd-pct", type=float, default=20.0)
    parser.add_argument("--min-profit-factor", type=float, default=1.2)
    parser.add_argument("--min-trades", type=int, default=20)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def summarize_period(trades: pd.DataFrame, period_label: str, min_trades: int, min_pf: float, max_dd: float) -> dict:
    if trades.empty:
        return {
            "period": period_label,
            "trades": 0,
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_trade_return_pct": 0.0,
            "avg_spread": 0.0,
            "pass": False,
            "notes": "No trades",
        }

    data = trades.copy().sort_values("exit_time")
    daily = data.groupby(data["exit_time"].dt.date).agg(daily_return_pct=("return_pct", "sum")).reset_index()
    daily["equity_pct"] = daily["daily_return_pct"].cumsum()
    daily["drawdown_pct"] = daily["equity_pct"] - daily["equity_pct"].cummax()

    gross_profit = data.loc[data["return_pct"] > 0, "return_pct"].sum()
    gross_loss = abs(data.loc[data["return_pct"] < 0, "return_pct"].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss else float("inf")
    return_pct = float(data["return_pct"].sum())
    max_dd_pct = float(daily["drawdown_pct"].min()) if not daily.empty else 0.0
    win_rate = float((data["return_pct"] > 0).mean())
    avg_trade_return_pct = float(data["return_pct"].mean())
    avg_spread = float(data["entry_spread"].mean())

    pass_rules = len(data) >= min_trades and return_pct > 0 and abs(max_dd_pct) <= max_dd and profit_factor >= min_pf
    notes = []
    if len(data) < min_trades:
        notes.append("low_trades")
    if return_pct <= 0:
        notes.append("negative_return")
    if abs(max_dd_pct) > max_dd:
        notes.append("dd_too_high")
    if profit_factor < min_pf:
        notes.append("pf_too_low")

    return {
        "period": period_label,
        "trades": int(len(data)),
        "return_pct": return_pct,
        "max_dd_pct": max_dd_pct,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "avg_trade_return_pct": avg_trade_return_pct,
        "avg_spread": avg_spread,
        "pass": bool(pass_rules),
        "notes": ",".join(notes) if notes else "ok",
    }


def main() -> None:
    args = parse_args()
    ensure_directories()
    trades_path = resolve_path(args.trades)
    strategy_path = resolve_path(args.strategy)
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing tick trades file: {trades_path}")

    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    trades["month"] = trades["exit_time"].dt.to_period("M").astype(str)
    trades["quarter"] = trades["exit_time"].dt.to_period("Q").astype(str)

    monthly = [
        summarize_period(group, period, args.min_trades, args.min_profit_factor, args.max_dd_pct)
        for period, group in trades.groupby("month")
    ]
    quarterly = [
        summarize_period(group, period, args.min_trades, args.min_profit_factor, args.max_dd_pct)
        for period, group in trades.groupby("quarter")
    ]
    overall = summarize_period(trades, "overall", args.min_trades, args.min_profit_factor, args.max_dd_pct)

    monthly_df = pd.DataFrame(monthly)
    quarterly_df = pd.DataFrame(quarterly)
    overall_df = pd.DataFrame([overall])

    monthly_path = RESULTS_DIR / f"{args.label}_monthly.csv"
    quarterly_path = RESULTS_DIR / f"{args.label}_quarterly.csv"
    overall_path = RESULTS_DIR / f"{args.label}_overall.csv"
    summary_path = RESULTS_DIR / f"{args.label}_summary.json"

    monthly_df.to_csv(monthly_path, index=False)
    quarterly_df.to_csv(quarterly_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    summary = {
        "strategy_file": str(strategy_path),
        "trades_file": str(trades_path),
        "overall": overall,
        "monthly_pass_rate": float(monthly_df["pass"].mean()) if not monthly_df.empty else 0.0,
        "quarterly_pass_rate": float(quarterly_df["pass"].mean()) if not quarterly_df.empty else 0.0,
        "failed_months": monthly_df.loc[~monthly_df["pass"], "period"].tolist() if not monthly_df.empty else [],
        "failed_quarters": quarterly_df.loc[~quarterly_df["pass"], "period"].tolist() if not quarterly_df.empty else [],
        "outputs": {
            "monthly": str(monthly_path),
            "quarterly": str(quarterly_path),
            "overall": str(overall_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({**summary, "summary_file": str(summary_path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
