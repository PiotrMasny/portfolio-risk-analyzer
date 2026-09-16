import numpy as np
import pandas as pd


def _validate_weights_length(
        weights: np.ndarray,
        n_assets: int
) -> None:
    if len(weights) != n_assets:
        raise ValueError("weights length must match number of assets")


def _validate_cov_matrix_is_square(
        cov_matrix: pd.DataFrame
) -> None:
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be square")


def portfolio_expected_return(
        weights: np.ndarray,
        mean_returns: pd.Series
) -> float:
    """
    Calculate the portfolio's expected return.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights in the same asset order as mean_returns.
    mean_returns : pd.Series
        Expected returns for each asset.

    Returns
    -------
    float
        Portfolio expected return.

    Raises
    ------
    ValueError
        If the number of weights does not match the number of assets.
    """
    _validate_weights_length(
        weights,
        len(mean_returns)
    )
    return float(weights @ mean_returns)


def portfolio_variance(
        weights: np.ndarray,
        cov_matrix: pd.DataFrame
) -> float:
    """
    Calculate the portfolio variance.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights in the same asset order as cov_matrix.
    cov_matrix : pd.DataFrame
        Covariance matrix between assets.

    Returns
    -------
    float
        Portfolio variance.

    Raises
    ------
    ValueError
        If the covariance matrix is not square or the number of weights
        does not match the number of assets.
    """
    _validate_cov_matrix_is_square(cov_matrix)

    _validate_weights_length(
        weights,
        cov_matrix.shape[0]
    )
    return float(weights @ cov_matrix @ weights)


def portfolio_volatility(
        weights: np.ndarray,
        cov_matrix: pd.DataFrame
) -> float:
    """
    Calculate the portfolio volatility.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights in the same asset order as cov_matrix.
    cov_matrix : pd.DataFrame
        Covariance matrix between assets.

    Returns
    -------
    float
        Portfolio volatility.

    Raises
    ------
    ValueError
        If the covariance matrix is not square or the number of weights
        does not match the number of assets.
    """
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))


def sharpe_ratio(
        weights: np.ndarray,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float
) -> float:
    """
    Calculate the portfolio Sharpe ratio.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights in the same asset order as mean_returns and cov_matrix.
    mean_returns : pd.Series
        Expected returns for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix between assets.
    risk_free_rate : float
        Risk-free rate expressed on the same basis as mean_returns.

    Returns
    -------
    float
        Portfolio Sharpe ratio.

    Raises
    ------
    ValueError
        If the covariance matrix is not square or the number of weights
        does not match the number of assets.
    """
    return float(
        (portfolio_expected_return(weights, mean_returns) - risk_free_rate)
        /
        portfolio_volatility(weights, cov_matrix)
    )


def portfolio_returns(
        returns: pd.DataFrame,
        weights: np.ndarray
) -> pd.Series:
    """
    Calculate portfolio returns for each observation.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns for each observation.
    weights: np.ndarray
        Portfolio weights in the same asset order as the columns of returns.

    Returns
    -------
    pd.Series
        Portfolio returns for each observation.

    Raises
    ------
    ValueError
        If the number of weights does not match the number of assets.
    """
    _validate_weights_length(
        weights,
        returns.shape[1]
    )
    return returns @ weights
