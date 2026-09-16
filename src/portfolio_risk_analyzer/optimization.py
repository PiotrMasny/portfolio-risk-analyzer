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

    num_of_assets = cov_matrix.shape[0]
    initial_weights = np.ones(num_of_assets) / num_of_assets

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=[(0, 1)] * num_of_assets,
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

    num_of_assets = cov_matrix.shape[0]
    initial_weights = np.ones(num_of_assets) / num_of_assets

    result = minimize(
        _negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=[(0, 1)] * num_of_assets,
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

    num_of_assets = cov_matrix.shape[0]

    if initial_weights is None:
        initial_weights = np.ones(num_of_assets) / num_of_assets

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=[(0, 1)] * num_of_assets,
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
