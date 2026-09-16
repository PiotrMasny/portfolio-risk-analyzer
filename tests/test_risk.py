import pandas as pd
import pytest

from portfolio_risk_analyzer.risk import (
    drawdown_series,
    historical_cvar,
    historical_var,
    max_drawdown,
)


@pytest.fixture
def returns():
    return pd.Series([0.10, -0.20, 0.05])


def test_drawdown_series(returns):
    pd.testing.assert_series_equal(
        drawdown_series(returns),
        pd.Series([0.00, -0.20, -0.16])
    )


def test_max_drawdown(returns):
    assert max_drawdown(returns) == pytest.approx(-0.20)


def test_drawdown_series_when_first_return_is_negative():
    returns = pd.Series([-0.10, 0.05])

    pd.testing.assert_series_equal(
        drawdown_series(returns),
        pd.Series([-0.10, -0.055])
    )


def test_historical_var():
    returns = pd.Series([
        -0.05, -0.04, -0.03, -0.02, -0.01,
        0.00, 0.01, 0.02, 0.03, 0.04,
        0.05, 0.06, 0.07, 0.08, 0.09,
        0.10, 0.11, 0.12, 0.13, 0.14,
        0.15
    ])

    result = historical_var(
        returns,
        confidence_level=0.95
    )

    assert result == pytest.approx(0.04)


def test_historical_cvar():
    returns = pd.Series([
        -0.05, -0.04, -0.03, -0.02, -0.01,
        0.00, 0.01, 0.02, 0.03, 0.04,
        0.05, 0.06, 0.07, 0.08, 0.09,
        0.10, 0.11, 0.12, 0.13, 0.14,
        0.15
    ])

    assert historical_cvar(
        returns,
        confidence_level=0.95
    ) == pytest.approx(0.045)
