# Portfolio Risk Analyzer

A Python-based portfolio optimization and risk analysis project using historical market data.

The project compares Minimum Variance and Maximum Sharpe portfolios under long-only constraints and evaluates their asset allocation, expected return, volatility, drawdown and historical tail risk.

## Features

- Historical market data download using `yfinance`
- Daily return calculation and annualization
- Minimum Variance Portfolio optimization
- Maximum Sharpe Portfolio optimization
- Efficient frontier construction
- Portfolio-level return and risk metrics
- Historical VaR and CVaR
- Maximum drawdown analysis
- Unit tests with `pytest`

## Project Structure

```text
portfolio-risk-analyzer/
├── src/
│   └── portfolio_risk_analyzer/
│       ├── data.py
│       ├── metrics.py
│       ├── optimization.py
│       ├── reporting.py
│       └── risk.py
├── tests/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_portfolio_analysis.ipynb
├── README.md
├── pyproject.toml
└── .gitignore
```

Core analytical logic is implemented in the `src/portfolio_risk_analyzer` package, while the `tests` directory contains unit tests for data processing, portfolio metrics, optimization and risk calculations.

The `notebooks` directory separates exploratory work from the final portfolio analysis workflow.

## Methodology

Adjusted closing prices are downloaded from Yahoo Finance and converted into daily arithmetic returns.

Expected returns and the covariance matrix are annualized using 252 trading days.

Portfolio optimization is performed under the following constraints:

- long-only positions,
- portfolio weights between 0% and 100%,
- portfolio weights sum to 100%.

Two optimized portfolios are considered:

- **Minimum Variance Portfolio** — minimizes portfolio variance.
- **Maximum Sharpe Portfolio** — maximizes expected excess return per unit of volatility.

The efficient frontier is constructed by minimizing portfolio variance for a range of target expected returns.

Portfolio risk is evaluated using annualized volatility, maximum drawdown, Historical VaR and Historical CVaR. VaR and CVaR are calculated from daily portfolio returns at a 95% confidence level.

The final notebook uses a 4% risk-free rate for Sharpe ratio calculations.

## Installation

Clone the repository:

```bash
git clone https://github.com/PiotrMasny/portfolio-risk-analyzer.git
cd portfolio-risk-analyzer
```

Create and activate a virtual environment, then install the project in editable mode:

```bash
pip install -e ".[dev]"
```

This installs the project together with its development dependencies, including pytest.


## Usage

The package can be used directly from Python to download market data, construct optimized portfolios and calculate portfolio risk metrics.

```python
from portfolio_risk_analyzer.data import (
    download_prices,
    calculate_returns,
    annualize_mean_returns,
    annualize_covariance
)
from portfolio_risk_analyzer.optimization import maximum_sharpe_weights
from portfolio_risk_analyzer.reporting import portfolio_summary

tickers = ['SPY', 'TLT', 'GLD', 'EEM', 'VNQ']

prices = download_prices(
    tickers,
    start='2020-01-01'
)

daily_returns = calculate_returns(prices)

annual_mean_returns = annualize_mean_returns(daily_returns)
annual_cov_matrix = annualize_covariance(daily_returns)

risk_free_rate = 0.04

weights = maximum_sharpe_weights(
    annual_mean_returns,
    annual_cov_matrix,
    risk_free_rate
)

summary = portfolio_summary(
    daily_returns,
    weights,
    annual_mean_returns,
    annual_cov_matrix,
    risk_free_rate
)

print(summary)
```

A complete analytical workflow, including portfolio comparison and visualizations, is available in:

`notebooks/02_portfolio_analysis.ipynb`

## Results

The analysis compares the Minimum Variance and Maximum Sharpe portfolios using:

- asset allocation,
- annualized expected return,
- annualized volatility,
- Sharpe ratio,
- maximum drawdown,
- Historical VaR,
- Historical CVaR,
- cumulative historical performance.

The final notebook also presents the long-only efficient frontier and compares the historical drawdown profiles of both optimized portfolios.

The results illustrate that minimizing volatility does not necessarily minimize other forms of risk. In the analyzed sample, the Minimum Variance Portfolio exhibited lower volatility and daily tail risk, while its maximum drawdown was deeper than that of the Maximum Sharpe Portfolio.

## Testing

The project includes unit tests for data processing, portfolio metrics, optimization, reporting and risk calculations.

Run the test suite with:

```bash
pytest
```

External market-data access is mocked in the relevant tests so that the test suite does not depend on a live Yahoo Finance connection.

## Limitations

The analysis is based on historical data and estimated expected returns and covariance matrices, which may not be representative of future market conditions.

Portfolio weights are estimated using the same historical sample that is later used for performance and risk analysis. The cumulative performance charts should therefore be interpreted as in-sample historical reconstructions rather than out-of-sample backtests.

The current implementation also assumes:

- no short selling,
- no transaction costs or taxes,
- no portfolio turnover constraints,
- no estimation error adjustments,
- a constant risk-free rate,
- constant target weights with daily rebalancing.

Historical VaR and CVaR are based on the empirical distribution of past daily returns and do not guarantee future loss limits.

## Future Improvements

Potential extensions include:

- out-of-sample and walk-forward portfolio evaluation,
- periodic portfolio rebalancing,
- transaction-cost modeling,
- covariance shrinkage,
- portfolio weight constraints,
- rolling risk metrics,
- dynamic risk-free rates,
- comparison with benchmark and equal-weight portfolios,
- additional risk and return estimation methods.

## Disclaimer

This project is intended for educational and analytical purposes only and does not constitute investment advice.