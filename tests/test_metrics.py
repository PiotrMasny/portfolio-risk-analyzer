import numpy as np
import pandas as pd
import pytest

from portfolio_risk_analyzer.metrics import (
    portfolio_expected_return,
    portfolio_returns,
    portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
)


@pytest.fixture
def weights():
    return np.array([0.5, 0.5])


@pytest.fixture
def mean_returns():
    return pd.Series(
        [0.10, 0.20],
        index=['A', 'B']
    )


@pytest.fixture
def cov_matrix():
    return pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09]
        ],
        index=['A', 'B'],
        columns=['A', 'B']
    )


@pytest.fixture
def risk_free_rate():
    return 0.05


@pytest.fixture
def invalid_weights():
    return np.array([0.2, 0.3, 0.5])


@pytest.fixture
def invalid_cov_matrix():
    return pd.DataFrame([
        [0.04, 0.01, 0.02],
        [0.01, 0.09, 0.03]
    ])


@pytest.fixture
def returns():
    return pd.DataFrame({
        'A': [0.10, -0.10],
        'B': [0.20, 0.00]
    })


def test_portfolio_expected_return(
        weights,
        mean_returns
):
    assert portfolio_expected_return(
        weights,
        mean_returns
    ) == pytest.approx(0.15)


def test_portfolio_expected_return_with_invalid_weights_length(
        invalid_weights,
        mean_returns
):
    with pytest.raises(ValueError):
        portfolio_expected_return(
            invalid_weights,
            mean_returns
        )


def test_portfolio_variance(
        weights,
        cov_matrix
):
    assert portfolio_variance(
        weights,
        cov_matrix
    ) == pytest.approx(0.0375)


def test_portfolio_variance_with_non_square_cov_matrix(
        weights,
        invalid_cov_matrix
):
    with pytest.raises(ValueError):
        portfolio_variance(
            weights,
            invalid_cov_matrix
        )


def test_portfolio_variance_with_invalid_weights_length(
        invalid_weights,
        cov_matrix
):
    with pytest.raises(ValueError):
        portfolio_variance(
            invalid_weights,
            cov_matrix
        )


def test_portfolio_volatility(
        weights,
        cov_matrix
):
    assert portfolio_volatility(
        weights,
        cov_matrix
    ) == pytest.approx(
        np.sqrt(0.0375)
    )


def test_sharpe_ratio(
        weights,
        mean_returns,
        cov_matrix,
        risk_free_rate
):
    assert (
            sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
            ==
            pytest.approx((0.15 - risk_free_rate) / np.sqrt(0.0375))
    )


def test_portfolio_returns(returns):

    weights = np.array([0.25, 0.75])

    expected = pd.Series([
        0.175,
        -0.025
    ])

    pd.testing.assert_series_equal(
        portfolio_returns(returns, weights),
        expected
    )


def test_portfolio_returns_with_invalid_weights_length(
        invalid_weights,
        returns
):
    with pytest.raises(ValueError):
        portfolio_returns(
            returns,
            invalid_weights
        )
