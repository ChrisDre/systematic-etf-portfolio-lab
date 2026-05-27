import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns from price data.
    """
    clean_prices = prices.dropna(axis=1, how="all")
    returns = clean_prices.pct_change().dropna()

    return returns

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized return.
    """
    compounded_growth = (1 + returns).prod()
    n_periods = len(returns)

    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized volatility.
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.
    """
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)

    return (ann_return - risk_free_rate) / ann_vol


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative returns from periodic returns.
    """
    return (1 + returns).cumprod() - 1


def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute drawdown series from returns.
    """
    cumulative = (1 + returns).cumprod()

    running_max = cumulative.cummax()

    drawdowns = cumulative / running_max - 1

    return drawdowns


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown.
    """
    return drawdown_series(returns).min()


def rolling_volatility(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Compute rolling annualized volatility.
    """
    return (
        returns.rolling(window).std()
        * np.sqrt(periods_per_year)
    )


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio.
    """
    rolling_return = (
        (1 + returns)
        .rolling(window)
        .apply(np.prod, raw=True)
    ) ** (periods_per_year / window) - 1

    rolling_vol = rolling_volatility(
        returns,
        window,
        periods_per_year,
    )

    return rolling_return / rolling_vol