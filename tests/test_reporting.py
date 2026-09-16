import numpy as np
import pandas as pd
import pytest

from portfolio_risk_analyzer.reporting import portfolio_summary


def test_portfolio_summary():

    daily_returns = pd.DataFrame({
        'A': [0.10, -0.20, 0.05],
        'B': [0.00, 0.10, 0.05]
    })

    weights = np.array([0.5, 0.5])

    annual_mean_returns = pd.Series({
        'A': 0.10,
        'B': 0.20
    })

    annual_cov_matrix = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09]
        ],
        index=['A', 'B'],
        columns=['A', 'B']
    )

    risk_free_rate = 0.05

    summary = portfolio_summary(
        daily_returns,
        weights,
        annual_mean_returns,
        annual_cov_matrix,
        risk_free_rate
    )

    assert summary['annual_return'] == pytest.approx(0.15)
    assert summary['annual_volatility'] == pytest.approx(np.sqrt(0.0375))
    assert summary['sharpe_ratio'] == pytest.approx(0.10 / np.sqrt(0.0375))
    assert summary['max_drawdown'] == pytest.approx(-0.05)
    assert summary['historical_var'] == pytest.approx(0.04)
    assert summary['historical_cvar'] == pytest.approx(0.05)
