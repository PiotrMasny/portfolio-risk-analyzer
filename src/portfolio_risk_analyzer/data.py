import pandas as pd
import yfinance as yf


def calculate_returns(
        prices: pd.DataFrame
) -> pd.DataFrame:

    return prices.pct_change(fill_method=None).dropna()


def annualize_mean_returns(
        returns: pd.DataFrame,
        periods_per_year: int = 252
) -> pd.Series:

    return returns.mean() * periods_per_year


def annualize_covariance(
        returns: pd.DataFrame,
        periods_per_year: int = 252
) -> pd.DataFrame:

    return returns.cov() * periods_per_year


def download_prices(
        tickers: list[str],
        start: str,
        end: str | None = None
) -> pd.DataFrame:

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True
    )

    return data['Close']
