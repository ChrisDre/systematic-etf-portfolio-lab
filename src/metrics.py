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