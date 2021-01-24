import pprint
import pandas
from util import AnalysisUtil, ModelUtil, ImportFromCSV, PlottingFunctions

# Extra settings
pandas.set_option("display.width", None)  # show all columns

"""Having first glance at the data"""
air_pollution = ImportFromCSV.import_parse_index_by_date("datasets/air_pollution.csv", "date")
# print(air_pollution.head())
air_pollution_base_stats = air_pollution.describe()  # base statistics of the data
# print(air_pollution_base_stats)
PlottingFunctions.print_multiple_columns_on_one_graph(air_pollution)
PlottingFunctions.print_and_save(air_pollution, "pollution_today", "results/pollution_today.png")

""" Automatic time series decomposition"""
decomp = AnalysisUtil.automatic_time_series_decomposition(air_pollution, "pollution_today", 365,
                                                          model="multiplicative")

PlottingFunctions.print_decomposition(decomp)

""" Trend analysis"""
trend = AnalysisUtil.trend_analysis(air_pollution, "pollution_today", 12)
PlottingFunctions.print_trend(trend_input=trend, decomp=decomp)

"""Seasonality"""
decomp = AnalysisUtil.automatic_time_series_decomposition(air_pollution, "pollution_today", -365,
                                                          model="multiplicative")
PlottingFunctions.print_decomposition(decomp)

"""Looking for weekly Seasonality"""

weekly_mean = AnalysisUtil.resample_weekly(air_pollution, "pollution_today")
PlottingFunctions.print_weekly_mean_series(weekly_mean)

"""Trying to find Seasonality manually (polynomial model)"""
curve = AnalysisUtil.polynomial_fit(air_pollution, "pollution_today", 100)
PlottingFunctions.print_polynomial_fit(air_pollution, "pollution_today", curve)

"""Checking for Noise in the data"""
noise = AnalysisUtil.noise_analysis(air_pollution, "pollution_today", 7, 30)
PlottingFunctions.print_noise_analysis(noise)

"""Stationarity analysis"""
PlottingFunctions.print_stationary_analysis(air_pollution, "pollution_today")

"""Rolling means and standard deviation analysis"""
param = AnalysisUtil.rolling_means_std_dev(air_pollution, "pollution_today", 12)
PlottingFunctions.print_roll_mean_std_dev(param)

# Time series data seems to be stationary

"""Augmented Dickey-Fuller test to confirm above"""

adf_test = AnalysisUtil.perform_aug_dickey_fuller(air_pollution, "pollution_today")
PlottingFunctions.print_aug_dickey_fuller(adf_test)

"""Splitting dataset to allow evaluation of the models"""

predictions_dict = {}
errors_dict = {}
df_training, df_test = ModelUtil.split_dataset(air_pollution, "2014-01-01")

"""Adding naive forecast (mean average value)"""

ModelUtil.calc_naive_mean(df_test, predictions_dict, errors_dict)

"""Adding yesterday's pollution"""
ModelUtil.calc_yesterday_result(df_test, predictions_dict, errors_dict)

"""--- Section of univariate time series forecasting models ---"""

"""SES (Simple Exponential Smoothing)"""
ModelUtil.calc_ses(df_training, df_test, air_pollution, "pollution_today", predictions_dict, errors_dict)

"""HWES (Holt Winter's Exponential Smoothing)"""
ModelUtil.calc_hwes(df_training, df_test, air_pollution, "pollution_today", predictions_dict, errors_dict)

"""Autoregression (AR)"""
ModelUtil.calc_ar(df_training, df_test, air_pollution, "pollution_today", predictions_dict, errors_dict)
PlottingFunctions.plot_save_model("AR", predictions_dict, df_test["pollution_today"])

"""Moving Average (MA)"""
ModelUtil.calc_ma(df_training, df_test, air_pollution, "pollution_today", predictions_dict, errors_dict)
PlottingFunctions.plot_save_model("MA", predictions_dict, df_test["pollution_today"])

"""Autoregressive Moving Average (ARMA)"""
ModelUtil.calc_arma(df_training, df_test, air_pollution, "pollution_today", predictions_dict, errors_dict)
PlottingFunctions.plot_save_model("ARMA", predictions_dict, df_test["pollution_today"])

"""Autoregressive integrated moving average (ARIMA)"""
ModelUtil.calc_arima(df_training, df_test, air_pollution, "pollution_today", predictions_dict, errors_dict)
PlottingFunctions.plot_save_model("ARIMA", predictions_dict, df_test["pollution_today"])

"""AutoARIMA"""
errors_dict, arima_key = ModelUtil.calc_auto_arima(df_training, df_test,
                                                   air_pollution, "pollution_today", predictions_dict, errors_dict)
PlottingFunctions.plot_save_model(arima_key, predictions_dict, df_test["pollution_today"])

"""Seasonal Autoregressive Integrated Moving-Average (SARIMA)"""
errors_dict, sarimax_key = ModelUtil.calc_sarima(df_training, df_test,
                                                 air_pollution, "pollution_today", predictions_dict, errors_dict)
PlottingFunctions.plot_save_model(sarimax_key, predictions_dict, df_test["pollution_today"])

"""AutoSARIMA"""

pprint.pprint(errors_dict)
