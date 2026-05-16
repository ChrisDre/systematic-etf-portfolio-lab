import pandas as pd


def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Compute trailing momentum as percentage price change over a lookback window.
    Example: 1 Year ago: 100, Today: 120 -> Momentum = 20% Trailing Return
    """
    return prices.pct_change(lookback)