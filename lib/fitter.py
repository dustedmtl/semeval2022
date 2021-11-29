"""Fit data."""

# pylint: disable=invalid-name, line-too-long

# from typing import List, Tuple, Dict
from typing import List, Tuple, Protocol
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ScikitModel(Protocol):
    # https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
    """Dummy typing class for sklearn."""

    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...


def get_fit(data, labels: List[str]) -> ScikitModel:
    """Get fitting with Logistic Regression."""
    logreg = LogisticRegression()
    logreg.fit(data, labels)
    return logreg


def get_fit_results(traindata, labels: List[str], data2, labels2: List[str]) -> Tuple[float, List[float], List[str]]:
    """Get predictions with Logistic Regression.."""
    fitter = get_fit(traindata, labels)
    score = fitter.score(data2, labels2)
    results = fitter.predict(data2)
    probs = fitter.predict_proba(data2)
    maxprobs = np.max(probs, axis=1)
    return score, maxprobs, results


def add_results(df: pd.DataFrame, predictions: List[str], gold: List[str]) -> pd.DataFrame:
    """Add results to dataframe."""
    newdf = df.copy()
    newdf['Label'] = gold['Label']
    newdf['Prediction'] = predictions
    newdf['Correct'] = newdf['Label'] == newdf['Prediction']
    # replace $ sign which messes up dataframe showing
    xy = newdf[['Previous', 'Target', 'Next']].replace({'\$': '$\$$'}, regex=True)
    newdf[['Previous', 'Target', 'Next']] = xy

    return newdf
