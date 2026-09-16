from unittest.mock import patch

import pandas as pd
import pytest

from portfolio_risk_analyzer.data import (
    annualize_covariance,
    annualize_mean_returns,
    calculate_returns,
    download_prices,
)


@pytest.fixture
def prices():
    return pd.DataFrame({
        'A': [100, 110, 121],
        'B': [200, 180, 198]
    })


def test_calculate_returns(prices):
    pd.testing.assert_frame_equal(
        calculate_returns(prices),
        pd.DataFrame(
            {
                'A': [0.1, 0.1],
                'B': [-0.1, 0.1]
            },
            index=[1, 2]
        )
    )


def test_annualize_mean_returns(prices):
    pd.testing.assert_series_equal(
        annualize_mean_returns(calculate_returns(prices), periods_per_year=10),
        pd.Series(
            {
                'A': 1.0,
                'B': 0.0
            }
        )
    )


def test_annualize_covariance(prices):
    returns = calculate_returns(prices)

    result = annualize_covariance(
        returns,
        periods_per_year=10
    )

    expected = pd.DataFrame(
        [
            [0.0, 0.0],
            [0.0, 0.2]
        ],
        index=['A', 'B'],
        columns=['A', 'B']
    )

    pd.testing.assert_frame_equal(
        result,
        expected
    )


def test_download_prices():
    test_df = pd.DataFrame(
        [
            [12.34, 300],
            [12.45, 280]
        ],
        index=pd.to_datetime([
            '2026-09-13',
            '2026-09-14'
        ]),
        columns=pd.MultiIndex.from_tuples([
            ('Close', 'SPY'),
            ('Volume', 'SPY')
        ])
    )

    with patch('portfolio_risk_analyzer.data.yf.download') as mock_download:
        mock_download.return_value = test_df

        result = download_prices(
            ['SPY'],
            start='2026-09-01',
            end='2026-09-15'
        )

        pd.testing.assert_frame_equal(
            result,
            test_df['Close']
        )

        mock_download.assert_called_once_with(
            ['SPY'],
            start='2026-09-01',
            end='2026-09-15',
            auto_adjust=True
        )
