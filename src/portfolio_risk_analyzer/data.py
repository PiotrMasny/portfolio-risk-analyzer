import pandas as pd
import yfinance as yf


def calculate_returns(
        prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate simple returns from asset prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset prices for each observation.

    Returns
    -------
    pd.DataFrame
        Asset returns for each observation.
    """
    return prices.pct_change(fill_method=None).dropna()


def annualize_mean_returns(
        returns: pd.DataFrame,
        periods_per_year: int = 252
) -> pd.Series:
    """
    Annualize mean asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns for each observation.
    periods_per_year : int
        Number of return observations per year.

    Returns
    -------
    pd.Series
        Annualized arithmetic mean return for each asset.
    """
    return returns.mean() * periods_per_year


def annualize_covariance(
        returns: pd.DataFrame,
        periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Annualize the asset return covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns for each observation.
    periods_per_year : int
        Number of return observations per year.

    Returns
    -------
    pd.DataFrame
        Annualized covariance matrix of asset returns.
    """
    return returns.cov() * periods_per_year


def download_prices(
        tickers: list[str],
        start: str,
        end: str | None = None
) -> pd.DataFrame:
    """
    Download adjusted closing prices for selected assets.

    Parameters
    ----------
    tickers : list[str]
        List of asset ticker symbols.
    start : str
        Start date for the requested data.
    end : str | None
        End date for the requested data.

    Returns
    -------
    pd.DataFrame
        Adjusted closing prices for the requested assets.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True
    )

    return data['Close']
