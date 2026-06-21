from __future__ import annotations


def calculate_position_size(
    account_balance: float,
    stop_loss_points: float,
    tick_value: float,
    risk_per_trade: float = 0.01,
) -> float:
    if account_balance <= 0:
        raise ValueError("account_balance must be positive")
    if stop_loss_points <= 0:
        raise ValueError("stop_loss_points must be positive")
    if tick_value <= 0:
        raise ValueError("tick_value must be positive")
    risk_amount = account_balance * risk_per_trade
    return risk_amount / (stop_loss_points * tick_value)


def daily_drawdown_limit(account_balance: float, limit: float = 0.03) -> float:
    return account_balance * limit
