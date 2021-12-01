"""Fit data."""

# pylint: disable=invalid-name, line-too-long

# from typing import List, Tuple, Dict
from typing import List, Tuple, Optional, Protocol
import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm.notebook import tqdm


class ScikitModel(Protocol):
    # https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
    """Dummy typing class for sklearn."""

    def fit(self, X, y, sample_weight=None): ...  # noqa: D102
    def predict(self, X): ...  # noqa: D102
    def predict_proba(self, X): ...  # noqa: D102
    def score(self, X, y, sample_weight=None): ...  # noqa: D102
    def set_params(self, **params): ...  # noqa: D102


def get_fit(data, labels: List[str], method: str = None) -> ScikitModel:
    """Get fitting with Logistic Regression or other method."""
    if not method:
        classifier = LogisticRegression()
    elif method == "rf":
        classifier = RandomForestClassifier()
    elif method == "knn":
        classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(data, labels)

    return classifier


def get_fit_results(traindata, labels: List[str],
                    data2, labels2: Optional[List[str]] = None,
                    method: Optional[str] = None) -> Tuple[float, List[float], List[str]]:
    """Get predictions with Logistic Regression.."""
    fitter = get_fit(traindata, labels, method)
    if labels2 is not None:
        score = fitter.score(data2, labels2)
    else:
        score = 0
    results = fitter.predict(data2)
    probs = fitter.predict_proba(data2)
    maxprobs = np.max(probs, axis=1)
    return score, maxprobs, results


def add_results(df: pd.DataFrame, predictions: List[str], gold: List[str]) -> pd.DataFrame:
    """Add results to dataframe."""
    newdf = df.copy()
    newdf['Label'] = gold['Label']  # type: ignore
    newdf['Prediction'] = predictions
    newdf['Correct'] = newdf['Label'] == newdf['Prediction']
    # replace $ sign which messes up dataframe showing
    xy = newdf[['Previous', 'Target', 'Next']].replace({'\$': '$\$$'}, regex=True)
    newdf[['Previous', 'Target', 'Next']] = xy

    return newdf


def get_trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Get trainable dataframe with float/int values."""
    res = df.copy()

    boolconvert = ['Quotes', 'Caps', 'Hassub', 'Trans']
    for col in boolconvert:
        if col in df.columns:
            res[col] = res[col].astype(bool)

    # cols = df.columns
    for _, dt in zip(enumerate(res.columns), res.dtypes):
        idx, col = _
        if idx < 7:
            # print("Dropping column", col)
            res.drop(col, axis=1, inplace=True)
        else:
            # print(idx, col, dt)
            if dt == 'float64':
                pass
            elif dt == 'bool':
                res[col] = res[col].astype(int)
                # print(idx, col, dt)
            else:
                res.drop(col, axis=1, inplace=True)

    res.reset_index()
    return res


def check_feats(train_data: pd.DataFrame, train_labels: List[str],
                dev_data: pd.DataFrame, dev_labels: List[str],
                method: Optional[str] = None,
                maxx=None):
    """Check performance with each of the features.."""
    cols = ['Label', 'Feats']
    cols.extend(train_data.columns)
    cols.extend(['Score'])

    resdf = pd.DataFrame([], columns=cols)
    for _ in train_data.columns:
        resdf[_] = resdf[_].astype(bool)

    perms = list(product([True, False], repeat=len(train_data.columns)))
    count = 0

    for p in tqdm(perms):
        count += 1
        if maxx and count > maxx:
            break
        if any(p):
            nu_train = train_data.copy()
            nu_dev = dev_data.copy()

            res = dict()
            rescols = []
            feats = 0
            for col, _drop in zip(train_data.columns, p):
                res[col] = _drop
                if _drop:
                    rescols.append(col)
                    feats += 1
                else:
                    nu_train = nu_train.drop(col, axis=1)
                    nu_dev = nu_dev.drop(col, axis=1)
            # print(nu_train.columns)
            classifier = get_fit(nu_train, train_labels, method)

            score = classifier.score(nu_dev, dev_labels)
            res['Label'] = ', '.join(rescols)
            res['Feats'] = feats
            res['Score'] = score
            resdf = resdf.append(res, ignore_index=True)
            # res['Label'] = lbl

    return resdf
