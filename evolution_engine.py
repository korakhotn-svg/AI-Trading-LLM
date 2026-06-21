from __future__ import annotations

import optuna

from backtest_engine import run_backtest


def evolve(candles, strategy_spec: dict, trials: int = 10) -> dict:
    # TODO: Expand the search space with entry filters, stop logic, and session windows.
    def objective(trial: optuna.Trial) -> float:
        candidate = dict(strategy_spec)
        candidate["atr_multiplier"] = trial.suggest_float("atr_multiplier", 1.0, 5.0)
        candidate["session_bias"] = trial.suggest_categorical("session_bias", ["auto", "london", "new_york", "overlap"])
        return run_backtest(candles, candidate)["fitness"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return {"best_params": study.best_params, "best_value": study.best_value}
