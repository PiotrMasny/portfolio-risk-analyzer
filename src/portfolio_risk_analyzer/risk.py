import pandas as pd


def drawdown_series(
        returns: pd.Series
) -> pd.Series:
    """
    Calculate the drawdown series from a sequence of returns.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns for each observation.

    Returns
    -------
    pd.Series
        Drawdown for each observation, expressed relative to the running cumulative
        wealth peak, with initial wealth normalized to 1.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax().clip(lower=1.0)

    return cumulative / running_max - 1


def max_drawdown(
        returns: pd.Series
) -> float:
    """
    Calculate the maximum drawdown from a sequence of returns.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns for each observation.

    Returns
    -------
    float
        Maximum drawdown as a negative return value.
    """
    return drawdown_series(returns).min()


def historical_var(
        returns: pd.Series,
        confidence_level: float = 0.95
) -> float:
    """
    Calculate historical Value at Risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns for each observation.
    confidence_level : float
        Confidence level used to determine the lower-tail return quantile.

    Returns
    -------
    float
        Historical VaR expressed as a loss magnitude.

    Raises
    ------
    ValueError
        If confidence_level is not between 0 and 1.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    return -returns.quantile(1 - confidence_level)


def historical_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
) -> float:
    """
    Calculate historical Conditional Value at Risk (CVaR).

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns for each observation.
    confidence_level : float
        Confidence level used to determine the lower-tail return threshold.

    Returns
    -------
    float
        Historical CVaR expressed as the average loss magnitude among
        returns at or below the VaR threshold.

    Raises
    ------
    ValueError
        If confidence_level is not between 0 and 1.
    """

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    threshold = returns.quantile(1 - confidence_level)

    return -returns[returns <= threshold].mean()
