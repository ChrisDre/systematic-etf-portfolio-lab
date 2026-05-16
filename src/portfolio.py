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
    return_weights: bool = False
) -> pd.Series:
    """
    Simulate a monthly rebalanced portfolio with intra-month weight drift.
    """
    target_weights = target_weights / target_weights.sum()
    target_weights = target_weights.reindex(returns.columns).fillna(0.0)

    portfolio_returns = []
    weights_history = []

    current_weights = target_weights.copy()

    previous_month = None

    for date, daily_returns in returns.iterrows():
        current_month = (date.year, date.month)

        # once new month, set weights back to target weights
        if current_month != previous_month:
            current_weights = target_weights.copy()
            previous_month = current_month
            
        weights_history.append(
            {"date": date, **current_weights.to_dict()}
        )
        portfolio_return = (current_weights * daily_returns).sum()
        portfolio_returns.append((date, portfolio_return))

        current_weights = current_weights * (1 + daily_returns)
        current_weights = current_weights / current_weights.sum()

    portfolio_returns = pd.Series(
        data=[r for _, r in portfolio_returns],
        index=[d for d, _ in portfolio_returns],
        name="portfolio_return",
    )

    weights_history = pd.DataFrame(weights_history)
    weights_history = weights_history.set_index("date")

    if return_weights:
        return portfolio_returns, weights_history

    return portfolio_returns