from pathlib import Path
import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of ETFs.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices = prices.dropna(how="all")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(save_path)

    return prices