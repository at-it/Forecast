import numpy as np
import pandas as pd
import sklearn.linear_model as sklm
import statsmodels.tsa.seasonal as sm
import statsmodels.tsa.stattools
from pandas.core.generic import NDFrame
from pandas.core.window import Window
from typing import Tuple
from classes.Trend import Trend
from classes.Noise import Noise


def automatic_time_series_decomposition(dataframe: pd.DataFrame, series_name: str, time: int,
                                        model: str) -> sm.DecomposeResult:
    """Perform decomposition using the Statsmodel framework on given series and time in days.
    Two model to be chosen from: additive/multiplicative."""
    if time < 0:
        series = dataframe[series_name][time:]
    else:
        series = dataframe[series_name][:time]
    result = sm.seasonal_decompose(series, model=model)
    return result


def trend_analysis(dataframe: pd.DataFrame, series_name: str, months: int) -> Trend:
    """Perform trend analysis using Sklearn based on panda DataFrame, series name and number of months."""
    series = dataframe[series_name]
    period = series.rolling(months).mean()
    X = [i for i in range(0, len(series))]
    X = np.reshape(X, (len(X), 1))
    y = series.values
    model = sklm.LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    return Trend(model, trend, series, period, months)


def resample_weekly(dataframe: pd.DataFrame, series_name: str) -> NDFrame:
    """Perform resampling of the Panda DataFrame on provided series name."""
    resample = dataframe.resample("W")
    weekly_mean = resample.mean()
    return weekly_mean[series_name]


def polynomial_fit(dataframe: pd.DataFrame, series_name: str, degree: int) -> list:
    """Perform polynomial fit on Panda DataFrame data for given series name and degree."""
    series = dataframe[series_name]
    series_val = series.values
    X = [i % 365 for i in range(0, len(series_val))]
    y = series_val
    coef = np.polyfit(X, y, degree)
    curve = []
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i] ** (degree - d) * coef[d]
        curve.append(value)
    return curve


def noise_analysis(dataframe: pd.DataFrame, series_name: str, num1: int, num2: int) -> Noise:
    """Perform noise analysis based on Pandas DataFrame object and series name. Two parameters to
    adjust length of window for rolling mean and rolling standard deviation."""
    series = dataframe[series_name]
    rol1 = series.rolling(window=num1).std()
    rol2 = series.rolling(window=num2).mean()
    return Noise(rol1, rol2, series, num1, num2)


def rolling_means_std_dev(dataframe: pd.DataFrame, series_name: str, num1: int) -> Tuple[Window, Window, pd.Series]:
    """Perform calculation of rolling mean and rolling standard deviation on given Pandas DataFrame object
    and series name."""
    series = dataframe[series_name]
    rolmean = series.rolling(window=num1).mean()
    rolstd = series.rolling(window=num1).std()
    return rolmean, rolstd, series


def perform_aug_dickey_fuller(dataframe: pd.DataFrame, series_name: str) -> Tuple[pd.Series, float, Window, Window]:
    """Perform Augmented Dickey Fuller test for data stationary on given Pandas DataFrame and series name."""
    series = dataframe[series_name]
    p_value = statsmodels.tsa.stattools.adfuller(series)[1]

    rolmean = rolling_means_std_dev(dataframe, series_name, 12)[0]
    rolstd = rolling_means_std_dev(dataframe, series_name, 12)[1]

    return series, p_value, rolmean, rolstd


