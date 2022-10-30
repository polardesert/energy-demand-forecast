import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from typing import Optional, List, Tuple
from typing import Dict, Any


__author__ = "Usman Ahmad"
__version__ = "1.0.1"


algo_map = {
    'log_reg': LogisticRegression(),
    'dt': DecisionTreeClassifier(),
    'knn': KNeighborsClassifier(),
    'gnb': GaussianNB(),
    'rf': RandomForestClassifier(),
    'dt_boosting': BaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=100,
        max_samples=0.8,
        oob_score=True,
        random_state=0),
    # 'svc': SVC(gamma='auto'),
    'vote_0': None,
    'vote_1': None,
    'vote_c': None,
}


def run_clf_mdl(
        data: pd.DataFrame,
        target: str,
        data_range: int = 10,
        fill_na: bool = False,
        algo_name: str = 'log_reg',
        test_size: float = 0.2,
        voting_params: Optional[List[Tuple]] = None,
        random_split: int = 1234,
) -> Dict[str, Any]:
    """
    run classification prediction on data
    :param test_size:
    :param random_split:
    :param fill_na:
    :param data:
    :param target:
    :param data_range:
    :param algo_name:
    :param voting_params:
    :return:
    """
    validate_inputs(algo_name, data, target)

    if fill_na:
        data.fillna(-99999, inplace=True)
    else:
        data.dropna(inplace=True)

    pred_target = target
    x_array = np.array(data.drop(pred_target, axis=1))
    y_array = np.array(data[pred_target])
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    x_latest = None
    if data_range:
        x_latest = np.array(data.drop(pred_target, axis=1).tail(data_range))

    x_train, x_test, y_train, y_test = train_test_split(
        x_array, y_array, test_size=test_size, random_state=random_split
    )

    # src = algo_map[algo_name]
    if "vote_" in algo_name:
        if voting_params:
            algo = VotingClassifier(estimators=voting_params)
        else:
            raise ValueError("Expecting voting parameters for voting regressor src")
    else:
        algo = algo_map[algo_name]

    algo.fit(x_train, y_train)
    acc = algo.score(x_test, y_test)
    y_predict = algo.predict(x_test)
    f1 = f1_score(y_test, y_predict)
    prec_score = precision_score(y_test, y_predict)
    rec_score = recall_score(y_test, y_predict)
    pred_category = algo.predict(x_latest)
    ham_loss = hamming_loss(y_test, y_predict)

    return {
        'preds': pred_category,
        'accuracy': acc,
        'f1_score': f1,
        'precision_score': prec_score,
        'recall_score': rec_score,
        'hamming_loss': ham_loss
    }


def validate_inputs(
        algo_name: str,
        data: pd.DataFrame,
        target: str,
        # pred_input: list
):
    """

    :param algo_name:
    :param data:
    :param target:
    :return:
    """

    if algo_name not in algo_map.keys():
        raise KeyError(f"{algo_name} not found in src map")

    if len(data) < 3:
        raise ValueError(
            "Data contains less than 3 rows of data points. Cannot "
            "proceed with regression run")

    if target not in data.columns:
        raise KeyError(f"Column {target} not found in data")
