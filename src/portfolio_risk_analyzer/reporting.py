import numpy as np
import pandas as pd

from portfolio_risk_analyzer.metrics import (
    portfolio_expected_return,
    portfolio_returns,
    portfolio_volatility,
    sharpe_ratio,
)
from portfolio_risk_analyzer.risk import historical_cvar, historical_var, max_drawdown


def portfolio_summary(
        daily_returns: pd.DataFrame,
        weights: np.ndarray,
        annual_mean_returns: pd.Series,
        annual_cov_matrix: pd.DataFrame,
        risk_free_rate: float,
        confidence_level: float = 0.95
) -> pd.Series:
    """
    Create a portfolio performance and risk summary.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Daily asset returns for each observation.
    weights : np.ndarray
        Portfolio weights in the same asset order as daily_returns,
        annual_mean_returns, and annual_cov_matrix.
    annual_mean_returns : pd.Series
        Annualized arithmetic mean return for each asset.
    annual_cov_matrix : pd.DataFrame
        Annualized covariance matrix of asset returns.
    risk_free_rate : float
        Risk-free rate expressed on the same basis as annual_mean_returns.
    confidence_level : float
        Confidence level used for historical VaR and CVaR calculations.

    Returns
    -------
    pd.Series
        Portfolio summary containing annual expected return, annual volatility,
        Sharpe ratio, maximum drawdown, and daily historical VaR and CVaR.
    """
    portfolio_daily_returns = portfolio_returns(
        daily_returns,
        weights
    )

    return pd.Series({
        'annual_return': portfolio_expected_return(
            weights,
            annual_mean_returns
        ),
        'annual_volatility': portfolio_volatility(
            weights,
            annual_cov_matrix
        ),
        'sharpe_ratio': sharpe_ratio(
            weights,
            annual_mean_returns,
            annual_cov_matrix,
            risk_free_rate
        ),
        'max_drawdown': max_drawdown(
            portfolio_daily_returns
        ),
        'historical_var': historical_var(
            portfolio_daily_returns,
            confidence_level
        ),
        'historical_cvar': historical_cvar(
            portfolio_daily_returns,
            confidence_level
        )
    })
