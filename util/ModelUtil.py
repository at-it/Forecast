import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from typing import Tuple
from util import EvaluationUtil


def split_dataset(dataframe: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and test set based on date provided in format YYYY-MM-DD"""
    df_training = dataframe.loc[dataframe.index <= split_date]
    df_test = dataframe.loc[dataframe.index > split_date]
    print(f"{len(df_training)} days of training data \n {len(df_test)} days of testing data")
    return df_training, df_test


def calc_naive_mean(testset: pd.DataFrame, pred_dict: dict, errors_dict: dict) -> dict:
    """Calculate naive mean on provided Pandas DataFrame. Results are stored into provided dicts
    with predictions and errors."""
    mean = testset.pollution_today.mean()
    mean = np.array([mean for u in range(len(testset.pollution_today))])
    pred_dict["Naive mean"] = mean
    actuals = testset.pollution_today
    hypothesis = pred_dict["Naive mean"]
    errors_dict = EvaluationUtil.evaluate_error("Naive mean", hypothesis, actuals, errors_dict)
    return errors_dict


def calc_yesterday_result(test_set: pd.DataFrame, pred_dict: dict, errors_dict: dict) -> dict:
    """Calculate using values of yesterday pollution. Results are stored into provided dicts
    with predictions and errors."""
    pred_dict["Yesterday's value"] = test_set.pollution_yesterday.values
    actuals = test_set.pollution_today
    hypothesis = pred_dict["Yesterday's value"]
    errors_dict = EvaluationUtil.evaluate_error("Yesterday's value", hypothesis, actuals, errors_dict)
    return errors_dict


def calc_ses(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
             series_name: str, pred_dict: dict, errors_dict: dict, ) -> dict:
    """Calculate using SES model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    yhat = list()
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = SimpleExpSmoothing(temp_train[series_name])
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train))
        yhat = yhat + [predictions]

    hypothesis = pd.concat(yhat)
    pred_dict["SES"] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error("SES", hypothesis, actuals, errors_dict)
    return errors_dict


def calc_hwes(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
              series_name: str, pred_dict: dict, errors_dict: dict) -> dict:
    """Calculate using HWES model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    yhat = list()
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = ExponentialSmoothing(temp_train[series_name])
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train))
        yhat = yhat + [predictions]

    hypothesis = pd.concat(yhat)
    pred_dict["HWES"] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error("HWES", hypothesis, actuals, errors_dict)
    return errors_dict


def calc_ar(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
            series_name: str, pred_dict: dict, errors_dict: dict, ) -> dict:
    """Calculate using AR model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    yhat = list()
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = AR(temp_train[series_name])
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    hypothesis = pd.concat(yhat)
    pred_dict["AR"] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error("AR", hypothesis, actuals, errors_dict)
    return errors_dict


def calc_ma(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
            series_name: str, pred_dict: dict, errors_dict: dict) -> dict:
    """Calculate using MA model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    yhat = list()
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = ARMA(temp_train[series_name], order=(0, 1))
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    hypothesis = pd.concat(yhat)
    pred_dict["MA"] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error("MA", hypothesis, actuals, errors_dict)
    return errors_dict


def calc_arma(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
              series_name: str, pred_dict: dict, errors_dict: dict) -> dict:
    """Calculate using ARMA model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    yhat = list()
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = ARMA(temp_train[series_name], order=(1, 1))
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    hypothesis = pd.concat(yhat)
    pred_dict["ARMA"] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error("ARMA", hypothesis, actuals, errors_dict)
    return errors_dict


def calc_arima(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
               series_name: str, pred_dict: dict, errors_dict: dict) -> dict:
    """Calculate using ARIMA model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    yhat = list()
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = ARIMA(temp_train[series_name], order=(1, 0, 0))
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    hypothesis = pd.concat(yhat)
    pred_dict["ARIMA"] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error("ARIMA", hypothesis, actuals, errors_dict)
    return errors_dict


def calc_auto_arima(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
                    series_name: str, pred_dict: dict, errors_dict: dict) -> Tuple[dict, str]:
    """Calculate using autoARIMA model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    automodel = pm.auto_arima(train_set[series_name], trace=True, error_action="ignore",
                              suppress_warnings="True", seasonal=False)
    automodel.fit(train_set[series_name])
    order = automodel.order
    yhat = list()
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = ARIMA(temp_train[series_name], order=order)
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    hypothesis = pd.concat(yhat)
    key = "AutoARIMA {}".format(order)
    pred_dict[key] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error(key, hypothesis, actuals, errors_dict)
    return errors_dict, key


def calc_sarima(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
                series_name: str, pred_dict: dict, errors_dict: dict) -> Tuple[dict, str]:
    """Calculate using SARIMA model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    yhat = list()
    order = (1, 0, 0)
    seasonal_order = (0, 0, 0, 3)
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = SARIMAX(dataframe[series_name], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        predicitions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predicitions]

    hypothesis = pd.concat(yhat)
    key = "SARIMAX {} {}".format(order, seasonal_order)
    pred_dict[key] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error(key, hypothesis, actuals, errors_dict)
    return errors_dict, key


def calc_autosarima(train_set: pd.DataFrame, test_set: pd.DataFrame, dataframe: pd.DataFrame,
                    series_name: str, pred_dict: dict, errors_dict: dict) -> Tuple[dict, str]:
    """Calculate using autoSARIMA model based on given training and test sets, on given Pandas DataFrame
    and series name. Results are stored into provided dicts with predictions and errors."""
    automodel = pm.auto_arima(train_set[series_name], trace=True, error_action="ignore",
                              suppress_warnings="True", seasonal=True, m=6, stepwise=True)
    automodel.fit(train_set[series_name])
    order = automodel.order
    seasonal_order = automodel.seasonal_order
    yhat = list()
    for t in tqdm(range(len(test_set[series_name]))):
        temp_train = dataframe[:len(train_set) + t]
        model = SARIMAX(temp_train[series_name], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        predictions = model_fit.predict(start=len(temp_train), end=len(temp_train), dynamic=False)
        yhat = yhat + [predictions]

    hypothesis = pd.concat(yhat)
    key = "AutoSARIMAX {} {}".format(order, seasonal_order)
    pred_dict[key] = hypothesis
    actuals = test_set[series_name]
    errors_dict = EvaluationUtil.evaluate_error(key, hypothesis, actuals, errors_dict)
    return errors_dict, key
