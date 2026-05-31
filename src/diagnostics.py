import pandas as pd


def momentum_selection_frequency(
    signals: pd.DataFrame,
    n_assets: int = 3,
) -> pd.Series:
    """
    Count how often each asset is selected
    by the momentum strategy.
    """
    counts = pd.Series(0, index=signals.columns)

    for _, signal in signals.iterrows():
        signal = signal.dropna()

        if len(signal) < n_assets:
            continue

        top_assets = signal.nlargest(n_assets).index

        counts.loc[top_assets] += 1

    return counts.sort_values(ascending=False)