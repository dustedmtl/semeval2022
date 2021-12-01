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
    """Get predictions with Logistic Regression."""
    fitter = get_fit(traindata, labels, method)
    if labels2 is not None:
        score = fitter.score(data2, labels2)
    else:
        score = 0
    results = fitter.predict(data2)
    probs = fitter.predict_proba(data2)
    maxprobs = np.max(probs, axis=1)
    return score, maxprobs, results


def get_fit_results2(traindata, labels: List[str],
                     devdata, devlabels: Optional[List[str]] = None,
                     method: Optional[str] = None,
                     byset: Optional[Tuple[List[bool], List[bool]]] = None) -> Tuple[float, List[float], List[str]]:
    """Get predictions with Logistic Regression."""
    if byset is not None:
        en_train_index, en_dev_index = byset
        # en_train_mask = np.array([False] * len(traindata))
        # for idx, val in enumerate(en_train_index):
        #     en_train_mask[idx] = val
        # en_dev_mask = np.array([False] * len(devdata))
        # for idx, val in enumerate(en_dev_index):
        #    en_dev_mask[idx] = val
        en_train_data = traindata.loc[en_train_index]
        en_train_labels = np.array(labels)[en_train_index]
        other_train_data = traindata.loc[~np.array(en_train_index)]
        other_train_labels = np.array(labels)[~np.array(en_train_index)]
        en_dev_data = devdata[en_dev_index]
        other_dev_data = devdata[~np.array(en_dev_index)]

        if devlabels is not None:
            en_dev_labels = np.array(devlabels)[en_dev_index]
            other_dev_labels = devlabels[~np.array(en_dev_index)]

        # print(len(traindata), len(devdata))
        # print(len(en_train_data))
        # print(len(other_train_data))
        # print(len(en_dev_data))
        # print(len(other_dev_data))

        en_fitter = get_fit(en_train_data, en_train_labels, method)
        if devlabels is not None:
            en_score = en_fitter.score(en_dev_data, en_dev_labels)
        else:
            en_score = 0
        en_results = en_fitter.predict(en_dev_data)
        en_probs = en_fitter.predict_proba(en_dev_data)
        en_maxprobs = np.max(en_probs, axis=1)

        other_fitter = get_fit(other_train_data, other_train_labels, method)
        if devlabels is not None:
            other_score = other_fitter.score(other_dev_data, other_dev_labels)
        else:
            other_score = 0

        other_results = other_fitter.predict(other_dev_data)
        other_probs = other_fitter.predict_proba(other_dev_data)
        other_maxprobs = np.max(other_probs, axis=1)

        res_ar = []
        prob_ar = []

        en_index = 0
        other_index = 0

        wt_score = (len(other_dev_data) * other_score + len(en_dev_data) * en_score) / len(devdata)

        for mask in en_dev_index:
            if mask:
                res_ar.append(en_results[en_index])
                prob_ar.append(en_maxprobs[en_index])
                en_index += 1
            else:
                res_ar.append(other_results[other_index])
                prob_ar.append(other_maxprobs[other_index])
                other_index += 1

        return wt_score, prob_ar, res_ar

    fitter = get_fit(traindata, labels, method)
    if devlabels is not None:
        score = fitter.score(devdata, devlabels)
    else:
        score = 0
    results = fitter.predict(devdata)
    probs = fitter.predict_proba(devdata)
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
    """Check performance with each of the features."""
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
            score, _probs, _results = get_fit_results(nu_train, train_labels, nu_dev, dev_labels, method)
            # classifier = get_fit(nu_train, train_labels, method)
            # score = classifier.score(nu_dev, dev_labels)
            res['Label'] = ', '.join(rescols)
            res['Feats'] = feats
            res['Score'] = score
            resdf = resdf.append(res, ignore_index=True)
            # res['Label'] = lbl

    return resdf


def multi_results(df: pd.DataFrame, gold: List[str],
                  labels1: List[str], scores1: List[str],
                  labels2: List[str], scores2: List[str],
                  lit_features: List[str] = None, idiom_features: List[str] = None) -> pd.DataFrame:
    """Get combined results from classifiers."""
    newdf = df.copy()
    newdf['Label'] = gold['Label']  # type: ignore
    newdf['Pred1'] = labels1
    newdf['Pred2'] = labels2
    newdf['Score1'] = scores1
    newdf['Score2'] = scores2

    agree = labels1 == labels2
    # print(np.sum(agree))
    # If the predictions agree, let's just use one
    newdf.loc[agree, 'Prediction'] = newdf['Pred1']
    # Disagreements
    newdf['Score1'] = newdf['Score1']
    use_pred1 = newdf['Score1'] > newdf['Score2']
    use_pred2 = ~use_pred1
    # print(np.sum(use_pred1), np.sum(use_pred2))
    disagree1 = ~agree & use_pred1
    disagree2 = ~agree & use_pred2
    # print(np.sum(disagree1), np.sum(disagree2))
    newdf.loc[disagree1, 'Prediction'] = newdf['Pred1']
    newdf.loc[disagree2, 'Prediction'] = newdf['Pred2']

    if lit_features:
        for f in lit_features:
            f_mask = newdf[f]
            newdf.loc[f_mask, 'Prediction'] = '1'

    if idiom_features:
        for f in idiom_features:
            f_mask = newdf[f]
            newdf.loc[f_mask, 'Prediction'] = '0'

    # replace $ sign which messes up dataframe showing
    xy = newdf[['Previous', 'Target', 'Next']].replace({'\$': '$\$$'}, regex=True)
    newdf[['Previous', 'Target', 'Next']] = xy

    return newdf
