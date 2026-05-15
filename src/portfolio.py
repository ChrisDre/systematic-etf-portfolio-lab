import pandas as pd


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Compute returns of an equal-weight portfolio.
    """
    n_assets = returns.shape[1]
    weights = pd.Series(1 / n_assets, index=returns.columns)

    portfolio_returns = returns @ weights

    return portfolio_returns


def weighted_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Compute weighted portfolio returns.
    """
    weights = weights / weights.sum()

    portfolio_returns = returns @ weights

    return portfolio_returns


def monthly_rebalance_with_drift(
    returns: pd.DataFrame,
    target_weights: pd.Series,
) -> pd.Series:
    """
    Simulate a monthly rebalanced portfolio with intra-month weight drift.
    """
    target_weights = target_weights / target_weights.sum()
    target_weights = target_weights.reindex(returns.columns).fillna(0.0)

    portfolio_returns = []
    current_weights = target_weights.copy()

    previous_month = None

    for date, daily_returns in returns.iterrows():
        current_month = (date.year, date.month)

        if current_month != previous_month:
            current_weights = target_weights.copy()
            previous_month = current_month

        portfolio_return = (current_weights * daily_returns).sum()
        portfolio_returns.append((date, portfolio_return))

        current_weights = current_weights * (1 + daily_returns)
        current_weights = current_weights / current_weights.sum()

    return pd.Series(
        data=[r for _, r in portfolio_returns],
        index=[d for d, _ in portfolio_returns],
        name="portfolio_return",
    )