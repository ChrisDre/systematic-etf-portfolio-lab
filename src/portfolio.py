import pandas as pd


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Compute returns of an equal-weight portfolio.
    """
    n_assets = returns.shape[1]
    weights = pd.Series(1 / n_assets, index=returns.columns)

    portfolio_returns = returns @ weights

    return portfolio_returns