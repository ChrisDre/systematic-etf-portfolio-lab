import pandas as pd
from src.costs import calculate_turnover, calculate_transaction_cost


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


def top_n_momentum_portfolio(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    n_assets: int = 3,
) -> pd.Series:
    """
    Allocate equally to the top-N assets by signal and hold for the next period.
    """
    signals = signals.reindex(returns.index)
    portfolio_returns = []

    for i in range(len(returns.index) - 1):
        signal_date = returns.index[i]
        return_date = returns.index[i + 1]

        signal = signals.loc[signal_date].dropna()

        if len(signal) < n_assets:
            continue

        top_assets = signal.nlargest(n_assets).index
        next_returns = returns.loc[return_date, top_assets]

        portfolio_return = next_returns.mean()

        portfolio_returns.append((return_date, portfolio_return))

    return pd.Series(
        data=[r for _, r in portfolio_returns],
        index=[d for d, _ in portfolio_returns],
        name="top_n_momentum",
    )


def monthly_top_n_momentum_portfolio(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    n_assets: int = 3,
) -> pd.Series:
    """
    Monthly top-N momentum strategy with intra-month weight drift.
    """
    signals = signals.reindex(returns.index)

    month_ends = returns.resample("ME").last().index
    month_ends = month_ends.intersection(returns.index)

    portfolio_returns = []

    for i in range(len(month_ends) - 1):
        signal_date = month_ends[i]
        next_month_end = month_ends[i + 1]

        signal = signals.loc[signal_date].dropna()

        if len(signal) < n_assets:
            continue

        top_assets = signal.nlargest(n_assets).index

        current_weights = pd.Series(0.0, index=returns.columns)
        current_weights.loc[top_assets] = 1 / n_assets

        period_returns = returns.loc[
            (returns.index > signal_date)
            & (returns.index <= next_month_end)
        ]

        for date, daily_returns in period_returns.iterrows():
            portfolio_return = (current_weights * daily_returns).sum()
            portfolio_returns.append((date, portfolio_return))

            current_weights = current_weights * (1 + daily_returns)
            current_weights = current_weights / current_weights.sum()

    return pd.Series(
        data=[r for _, r in portfolio_returns],
        index=[d for d, _ in portfolio_returns],
        name="monthly_top_n_momentum",
    )

def monthly_top_n_momentum_portfolio_with_costs(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    n_assets: int = 3,
    cost_rate: float = 0.001,
    return_weights: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monthly top-N momentum strategy with weight drift, turnover, and transaction costs.

    At each month-end:
    - select top-N assets by momentum
    - calculate turnover from previous weights to new target weights
    - apply transaction cost on first trading day of next month
    - let weights drift during the month
    """
    signals = signals.reindex(returns.index)

    month_ends = returns.resample("ME").last().index
    month_ends = month_ends.intersection(returns.index)

    results = []
    weights_history = []

    previous_weights = pd.Series(0.0, index=returns.columns)

    for i in range(len(month_ends) - 1):
        signal_date = month_ends[i]
        next_month_end = month_ends[i + 1]

        signal = signals.loc[signal_date].dropna()

        if len(signal) < n_assets:
            continue

        top_assets = signal.nlargest(n_assets).index

        target_weights = pd.Series(0.0, index=returns.columns)
        target_weights.loc[top_assets] = 1 / n_assets

        turnover = calculate_turnover(previous_weights, target_weights)
        transaction_cost = calculate_transaction_cost(turnover, cost_rate)

        period_returns = returns.loc[
            (returns.index > signal_date)
            & (returns.index <= next_month_end)
        ]

        current_weights = target_weights.copy()

        for j, (date, daily_returns) in enumerate(period_returns.iterrows()):
            gross_return = (current_weights * daily_returns).sum()

            cost = transaction_cost if j == 0 else 0.0
            net_return = gross_return - cost

            results.append({
                "date": date,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover if j == 0 else 0.0,
                "transaction_cost": cost,
            })

            weights_history.append({
                "date": date,
                **current_weights.to_dict(),
            })

            current_weights = current_weights * (1 + daily_returns)
            current_weights = current_weights / current_weights.sum()

        previous_weights = current_weights.copy()

    results = pd.DataFrame(results).set_index("date")
    weights_history = pd.DataFrame(weights_history).set_index("date")

    if return_weights:
        return results, weights_history

    return results