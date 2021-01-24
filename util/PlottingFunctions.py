import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import statsmodels.tsa.api as smt
import statsmodels.tsa.seasonal as sm
from pandas.core.window import Window
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from classes.Noise import Noise
from classes.Trend import Trend

plt.style.use("seaborn-darkgrid")
deep_green_blue = (44 / 255, 63 / 255, 59 / 255, 1)


def print_multiple_columns_on_one_graph(dataframe: pd.DataFrame) -> None:
    """Plot multiple columns on one graph based on provided Panda DataFrame"""
    values = dataframe.values
    groups = dataframe.columns.size
    i = 1
    for group in range(0, groups):
        plt.tight_layout(pad=0.5)
        plt.subplot(groups, 1, i)
        plt.plot(values[:, group])
        plt.title(dataframe.columns[group], y=0.25, loc="right")
        i += 1
    plt.show()


def print_and_save(dataframe: pd.DataFrame, series_name: str, filename: str) -> None:
    """Print and save series from given Panda DataFrame to specified path existing in current directory."""
    plt.figure(num=None, figsize=(30, 10), dpi=80, facecolor="w", edgecolor="k")  # num=None as no name was given
    plt.title(series_name, fontsize=30),
    series_name = pd.Series.get(dataframe, series_name)
    plt.plot(series_name)
    plt.savefig(filename)
    plt.show()


def print_weekly_mean_series(series_name: pd.Series) -> None:
    """Print series based on provided panda Dataframe and series name."""
    series_name.plot(label="Weekly mean")
    plt.title("Resampled series to weekly mean values")
    plt.legend()
    plt.show()


def print_polynomial_fit(dataframe: pd.DataFrame, series_name: str, curve: list) -> None:
    """Plot polynomial fit of the data given by provided curve for Panda DataFrame and series name."""
    series = pd.Series.get(dataframe, series_name)
    series_val = series.values
    plt.plot(series_val, label="Original")
    plt.plot(curve, color=deep_green_blue, linewidth=3.0, label="polynomial model")
    plt.legend()
    plt.title("Polynomial fit to fit seasonality")
    plt.show()


def print_decomposition(decomp: sm.DecomposeResult) -> None:
    """Plot decomposition provided by StatsModel DecomposeResult."""
    pylab.rcParams['figure.figsize'] = (18, 8)
    decomp.plot()
    plt.show()


def print_decomposition_series(decomp: sm.DecomposeResult, dataframe: pd.DataFrame, series_name: str) -> None:
    """Plot decomposition series provided by StatsModel DecomposeResult."""
    pylab.rcParams["figure.figsize"] = (18, 8)
    plt.figure(num=None, figsize=(40, 20), dpi=80, facecolor="w", edgecolor="k")
    decomp.plot()
    plt.show()


def print_trend(trend_input: Trend, decomp: sm.DecomposeResult) -> None:
    """Plot trend provided as a result of DecomposeResult from StatsModel."""
    fig = plt.figure(figsize=(15, 7))
    layout = (3, 2)
    pm_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    mv_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
    fit_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
    pm_ax.plot(decomp.trend)
    pm_ax.set_title("Automatic decomposed trend")
    mm = trend_input.period
    mv_ax.plot(mm)
    mv_ax.set_title("Moving average %d steps" % trend_input.months)
    X = [i for i in range(0, len(trend_input.series))]
    X = np.reshape(X, (len(X), 1))
    y = trend_input.series.values
    fit_ax.plot(trend_input.trend)
    fit_ax.set_title("Trend fitted by Linear Regression")
    plt.tight_layout()
    plt.show()


def print_noise_analysis(noise: Noise) -> None:
    """Print results of noise analysis, mainly autocorrelation, standard deviation histogram and mean over time."""
    plt.figure(figsize=(12, 7))
    layout = (2, 2)
    hist_ax = plt.subplot2grid(layout, (0, 0))
    ac_ax = plt.subplot2grid(layout, (1, 0))
    hist_std_ax = plt.subplot2grid(layout, (0, 1))
    mean_ax = plt.subplot2grid(layout, (1, 1))

    noise.series.hist(ax=hist_ax)
    hist_ax.set_title("Original series histogram")

    series_val = noise.series.values
    plot_acf(series_val, lags=noise.num2, ax=ac_ax)
    ac_ax.set_title("Autocorrelation")

    noise.rol1.hist(ax=hist_std_ax)
    hist_std_ax.set_title("Standard deviation histogram")

    noise.rol2.plot(ax=mean_ax)
    mean_ax.set_title("Mean over time")

    plt.show()


def print_stationary_analysis(dataframe: pd.DataFrame, series_name: str) -> None:
    """Print Autocorrelation and Partial Auto Correlation graphs based on provided Pandas DataFrame
    and series name."""
    series = dataframe[series_name]
    plot_acf(series, lags=30)
    plot_pacf(series, lags=30)
    plt.show()


def print_roll_mean_std_dev(param: (Window, Window, pd.Series)) -> None:
    """Plot rolling mean and rolling std dev of given Pandas DataFrame object."""
    rolmean = param[0]
    rolstd = param[1]
    series = param[2]

    orig = plt.plot(series, label="Original")
    mean = plt.plot(rolmean, color="yellow", label="Rolling Mean")
    std = plt.plot(rolstd, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard Deviation")
    plt.show()


def print_aug_dickey_fuller(param: (pd.DataFrame, float, Window, Window), lags=None) -> None:
    """Print results of Augemented Dickey Fuller test for data stationary based on Pandas DataFrame object,
    p-value, rolling mean and rolling std dev."""
    series = param[0]
    p_value = param[1]
    rolmean = param[2]
    rolstd = param[3]

    fig = plt.figure(figsize=(12, 7))
    layout = (3, 2)
    test_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    mean_std_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
    series.plot(ax=test_ax)

    hypothesis_result = "Data assumed as stationary" if p_value <= 0.05 else "Data is not stationary."
    test_ax.set_title("Time series stationary analysis Plots\n Dickey-Fuller: p={0:.5f} Result: {1}"
                      .format(p_value, hypothesis_result))
    smt.graphics.plot_acf(series, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(series, lags=lags, ax=pacf_ax)
    plt.tight_layout()

    orig = plt.plot(series, label="Original")
    mean = plt.plot(rolmean, color="yellow", label="Rolling Mean")
    std = plt.plot(rolstd, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & Standard deviation")
    plt.show()


def plot_save_model(model_name: str, pred_dict: dict, actuals: pd.DataFrame) -> None:
    """Plot model provided in Pandas DataFrame object."""
    filename = "results/{}.png".format(model_name)
    plt.plot(actuals, label="Actuals")
    plt.plot(pred_dict[model_name], color="yellow", label=model_name)
    plt.legend()
    plt.savefig(filename)
    plt.show()
