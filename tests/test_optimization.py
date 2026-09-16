import numpy as np
import pandas as pd
import pytest

from portfolio_risk_analyzer.optimization import (
    efficient_frontier,
    maximum_sharpe_weights,
    minimum_variance_weights,
    minimum_variance_weights_for_return,
)


@pytest.fixture
def cov_matrix():
    return pd.DataFrame(
        [
            [0.04, 0.00],
            [0.00, 0.09]
        ],
        index=['A', 'B'],
        columns=['A', 'B']
    )


@pytest.fixture
def mean_returns():
    return pd.Series(
        [0.10, 0.20],
        index=['A', 'B']
    )


@pytest.fixture
def risk_free_rate():
    return 0.05


@pytest.fixture
def target_return():
    return 0.17


def test_minimum_variance_weights(cov_matrix):
    weights = minimum_variance_weights(cov_matrix)

    assert weights.sum() == pytest.approx(1.0)

    np.testing.assert_allclose(
        weights,
        np.array([0.6923, 0.3077]),
        atol=1e-4
    )


def test_maximum_sharpe_weights(mean_returns, cov_matrix, risk_free_rate):
    weights = maximum_sharpe_weights(mean_returns, cov_matrix, risk_free_rate)

    assert weights.sum() == pytest.approx(1.0)

    np.testing.assert_allclose(
        weights,
        np.array([0.4286, 0.5714]),
        atol=1e-4
    )


def test_minimum_variance_weights_for_return(mean_returns, cov_matrix, target_return):
    weights = minimum_variance_weights_for_return(mean_returns, cov_matrix, target_return)

    assert weights.sum() == pytest.approx(1.0)

    assert weights @ mean_returns == pytest.approx(target_return)

    np.testing.assert_allclose(
        weights,
        np.array([0.3, 0.7]),
        atol=1e-4
    )


def test_minimum_variance_for_unreachable_return(mean_returns, cov_matrix):
    with pytest.raises(RuntimeError):
        minimum_variance_weights_for_return(mean_returns, cov_matrix, 0.30)


def test_efficient_frontier(mean_returns, cov_matrix):
    n_points = 10

    frontier = efficient_frontier(mean_returns, cov_matrix, n_points=n_points)

    assert len(frontier) == n_points

    assert list(frontier.columns) == ['annual_return', 'annual_volatility']

    assert frontier['annual_return'].is_monotonic_increasing

    assert frontier['annual_return'].iloc[-1] == pytest.approx(mean_returns.max())

    assert frontier['annual_volatility'].iloc[0] == pytest.approx(frontier['annual_volatility'].min())

    assert np.all(
        np.diff(frontier['annual_volatility']) >= -1e-6
    )
