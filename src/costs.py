import pandas as pd


def calculate_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series,
) -> float:
    """
    Calculate one-way portfolio turnover.

    Turnover = 0.5 * sum(abs(new_weights - old_weights))
    """
    all_assets = old_weights.index.union(new_weights.index)

    old_weights = old_weights.reindex(all_assets).fillna(0.0)
    new_weights = new_weights.reindex(all_assets).fillna(0.0)

    turnover = 0.5 * (new_weights - old_weights).abs().sum()

    return turnover


def calculate_transaction_cost(
    turnover: float,
    cost_rate: float = 0.001,
) -> float:
    """
    Calculate transaction cost from turnover.

    cost_rate=0.001 means 10 basis points.
    """
    return turnover * cost_rate