import numpy as np
import pandas as pd


def portfolio_expected_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    return float(weights @ mean_returns)


def portfolio_variance(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return float(weights @ cov_matrix @ weights)


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return float(np.sqrt(weights @ cov_matrix @ weights))


def sharpe_ratio(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
    return float((portfolio_expected_return(weights, mean_returns) - risk_free_rate) / portfolio_volatility(weights, cov_matrix))


def portfolio_returns(
        returns: pd.DataFrame,
        weights: np.ndarray
) -> pd.Series:

    return returns @ weights
