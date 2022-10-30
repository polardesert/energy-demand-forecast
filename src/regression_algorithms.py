#!/usr/bin/env python
"""
Regression ML algos
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet  #
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, HistGradientBoostingRegressor
)
from typing import Optional, List, Tuple
from typing import Dict, Any

__author__ = "Usman Ahmad"
__version__ = "1.0.1"

algo_map = {
    'lin_reg': LinearRegression(n_jobs=100),
    'svr_linear': svm.SVR(),
    'svr_poly': svm.SVR(kernel="poly"),
    'svr_poly_auto': svm.SVR(kernel="poly", gamma='auto'),
    'svr_poly_scale': svm.SVR(kernel="poly", gamma='scale'),
    'svr_poly_lerr': svm.SVR(kernel="poly", C=0.1),
    'svr_poly_auto_lerr': svm.SVR(kernel="poly", gamma='auto', C=0.1),
    'svr_poly_scale_lerr': svm.SVR(kernel="poly", gamma='scale', C=0.1),
    'svr_poly_coef': svm.SVR(kernel="poly", coef0=1),
    'svr_poly_auto_coef': svm.SVR(kernel="poly", gamma='auto', coef0=1),
    'svr_poly_scale_coef': svm.SVR(kernel="poly", gamma='scale', coef0=1),
    'svr_poly_lerr_coef': svm.SVR(kernel="poly", C=0.1, coef0=1),
    'svr_poly_auto_lerr_coef': svm.SVR(kernel="poly", gamma='auto', C=0.1, coef0=1),
    'svr_poly_scale_lerr_coef': svm.SVR(
        kernel="poly", gamma='scale', C=0.1, coef0=1
    ),
    'lasso': Lasso(alpha=0.1),
    'elasticnet': ElasticNet(random_state=0),
    'ridge': Ridge(alpha=0.5),
    'randomforest_0': RandomForestRegressor(max_depth=2, random_state=0),
    'grad_boost_0': GradientBoostingRegressor(random_state=0),
    'grad_boost_0_LR': GradientBoostingRegressor(random_state=0, learning_rate=.01),
    'hist_grad_boost_0': HistGradientBoostingRegressor(random_state=0),
    'hist_grad_boost_0_LR': HistGradientBoostingRegressor(
        random_state=0,
        learning_rate=.01
        ),
    'randomforest_1': RandomForestRegressor(max_depth=2, random_state=1),
    'grad_boost_1': GradientBoostingRegressor(random_state=1),
    'vote_0': None,
    'vote_1': None,
    'vote_c': None,
}


def run_reg_pred(
        data: pd.DataFrame,
        target: str,
        pred_range: int,
        fill_na: bool = False,
        algo_name: str = "lin_reg",
        test_size: float = 0.2,
        voting_params: Optional[List[Tuple]] = None,
) -> Dict[str, Any]:
    """
    run regression prediction on data_regression
    :param voting_params:
    :param test_size:
    :param fill_na:
    :param data:
    :param target:
    :param pred_range:
    :param algo_name:
    :return:
    """
    validate_inputs(algo_name, data, target, pred_range)

    if fill_na:
        data.fillna(-99999, inplace=True)
    else:
        data.dropna(inplace=True)

    pred_target = f"pred_{target}"
    data[pred_target] = data[target].shift(-pred_range)
    x_array = np.array(data.drop(pred_target, axis=1))
    x_array = preprocessing.scale(x_array)
    x_latest = x_array[-pred_range:]
    x_array = x_array[:-pred_range]
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    y_array = np.array(data[pred_target])

    x_train, x_test, y_train, y_test = train_test_split(
        x_array, y_array, test_size=test_size
    )

    if "vote_" in algo_name:
        if voting_params is not None:
            algo = VotingRegressor(estimators=voting_params)
        else:
            raise ValueError("Expecting voting parameters for voting regressor src")
    else:
        algo = algo_map[algo_name]

    algo.fit(x_train, y_train)
    acc = algo.score(x_test, y_test)
    pred_set = algo.predict(x_latest)
    mse = mean_squared_error(y_test, algo.predict(x_test))

    return {
        "preds": pred_set,
        "accuracy": acc,
        "mse": mse
    }


def validate_inputs(
        algo_name: str,
        data: pd.DataFrame,
        target: str,
        pred_range: int
) -> None:
    """

    :param pred_range:
    :param algo_name:
    :param data:
    :param target:
    :return:
    """
    if target not in data.columns:
        raise KeyError(f"Column {target} not found in data")
    if len(data) < 3:
        raise ValueError(
            "Data contains less than 3 rows of data_regression points. Cannot "
            "proceed with regression run"
        )
    if algo_name not in algo_map.keys():
        raise KeyError(f"{algo_name} not found in src map")
    if pred_range < 1:
        raise ValueError("Prediction range cannot be 0 or negative")
