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
