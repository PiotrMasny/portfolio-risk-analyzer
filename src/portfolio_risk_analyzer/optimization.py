import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_risk_analyzer.metrics import (
    portfolio_expected_return,
    portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
)


def minimum_variance_weights(
        cov_matrix: pd.DataFrame
) -> np.ndarray:
    """
    Calculate the long-only minimum-variance portfolio weights.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns.

    Returns
    -------
    np.ndarray
        Portfolio weights that minimize variance, subject to weights
        summing to 1 and each weight being between 0 and 1.

    Raises
    ------
    RuntimeError
        If the numerical optimization fails.
    """
    n_assets = cov_matrix.shape[0]
    initial_weights = np.ones(n_assets) / n_assets

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=[(0, 1)] * n_assets,
        constraints={
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1
        }
    )

    if not result.success:
        raise RuntimeError(
            f"Variance optimization failed: {result.message}"
        )

    return result.x


def _negative_sharpe_ratio(
        weights: np.ndarray,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float
) -> float:
    """
    Calculate the negative portfolio Sharpe ratio.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights in the same asset order as mean_returns and cov_matrix.
    mean_returns : pd.Series
        Expected returns for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns, expressed on the same basis
        as mean_returns.
    risk_free_rate : float
        Risk-free rate expressed on the same basis as mean_returns.

    Returns
    -------
    float
        Negative portfolio Sharpe ratio.
    """
    return -sharpe_ratio(
        weights,
        mean_returns,
        cov_matrix,
        risk_free_rate
    )


def maximum_sharpe_weights(
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float
) -> np.ndarray:
    """
    Calculate the long-only maximum-Sharpe portfolio weights.

    Parameters
    ----------
    mean_returns : pd.Series
        Expected returns for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns in the same asset order as
        mean_returns and on the same basis.
    risk_free_rate : float
        Risk-free rate expressed on the same basis as mean_returns.

    Returns
    -------
    np.ndarray
        Portfolio weights that maximize the Sharpe ratio, subject to weights
        summing to 1 and each weight being between 0 and 1.

    Raises
    ------
    RuntimeError
        If the numerical optimization fails.
    """
    n_assets = cov_matrix.shape[0]
    initial_weights = np.ones(n_assets) / n_assets

    result = minimize(
        _negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=[(0, 1)] * n_assets,
        constraints={
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1
        }
    )

    if not result.success:
        raise RuntimeError(
            f"Sharpe ratio optimization failed: {result.message}"
        )

    return result.x


def minimum_variance_weights_for_return(
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: float,
        initial_weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Calculate minimum-variance portfolio weights for a target return.

    Parameters
    ----------
    mean_returns : pd.Series
        Expected returns for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns in the same asset order as
        mean_returns and on the same basis.
    target_return : float
        Required portfolio expected return, expressed on the same basis
        as mean_returns.
    initial_weights : np.ndarray | None
        Starting portfolio weights for the numerical optimizer. If None,
        equal weights are used.

    Returns
    -------
    np.ndarray
        Portfolio weights that minimize variance while achieving the target
        return, subject to long-only and fully invested constraints.

    Raises
    ------
    RuntimeError
        If the numerical optimization fails.
    """
    n_assets = cov_matrix.shape[0]

    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=[(0, 1)] * n_assets,
        constraints=[
            {
                'type': 'eq',
                'fun': lambda weights: np.sum(weights) - 1
            },
            {
                'type': 'eq',
                'fun': lambda weights: weights @ mean_returns - target_return
            }
        ]
    )

    if not result.success:
        raise RuntimeError(
            f"Variance for return optimization failed: {result.message}"
        )

    return result.x


def efficient_frontier(
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        n_points: int = 50
) -> pd.DataFrame:
    """
    Calculate the long-only efficient frontier.

    Parameters
    ----------
    mean_returns : pd.Series
        Expected annual returns for each asset.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix of asset returns.
    n_points : int
        Number of portfolios used to approximate the efficient frontier.

    Returns
    -------
    pd.DataFrame
        Efficient-frontier portfolios with annual expected return and
        annual volatility.

    Raises
    ------
    RuntimeError
        If any portfolio optimization fails.
    ValueError
        If n_points is less than 2.

    Notes
    -----
    The frontier is generated from the minimum-variance portfolio to the
    maximum expected return under long-only constraints. Each optimization
    uses the previous portfolio weights as the initial solution.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    min_var_weights = minimum_variance_weights(cov_matrix)
    min_var_return = portfolio_expected_return(min_var_weights, mean_returns)
    max_return = mean_returns.max()

    target_returns = np.linspace(
        min_var_return,
        max_return,
        n_points
    )

    results = []

    initial_weights = min_var_weights

    for target_return in target_returns:
        weights = minimum_variance_weights_for_return(
            mean_returns,
            cov_matrix,
            target_return,
            initial_weights=initial_weights
        )

        initial_weights = weights

        results.append({
            'annual_return': portfolio_expected_return(weights, mean_returns),
            'annual_volatility': portfolio_volatility(weights, cov_matrix)
        })

    return pd.DataFrame(results)
