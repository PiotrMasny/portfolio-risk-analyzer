import pandas as pd


def drawdown_series(
        returns: pd.Series
) -> pd.Series:

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax().clip(lower=1.0)

    return cumulative / running_max - 1


def max_drawdown(
        returns: pd.Series
) -> float:
    return drawdown_series(returns).min()


def historical_var(
        returns: pd.Series,
        confidence_level: float = 0.95
) -> float:
    return - returns.quantile(1 - confidence_level)


def historical_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
) -> float:

    threshold = returns.quantile(1 - confidence_level)

    return - returns[returns <= threshold].mean()
