import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def evaluate_error(model: str, hypothesis: np.array, actuals: pd.Series, errors: dict):
    if not isinstance(errors, dict):
        errors = dict()
    MAE = tf.metrics.MAE(actuals, hypothesis)
    RMSE = tf.metrics.RootMeanSquaredError()
    RMSE.update_state(actuals, hypothesis)
    MAPE = tf.metrics.MAPE(actuals, hypothesis)
    R2 = r2_score(actuals, hypothesis)
    errors[model] = {"MAE": MAE.numpy(), "RMSE": RMSE.result().numpy(), "MAPE": MAPE.numpy(), "R2": R2}
    return errors
