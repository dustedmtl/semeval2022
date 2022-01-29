"""Fit data."""

# pylint: disable=invalid-name, line-too-long, unused-variable

# from typing import List, Tuple, Dict
from typing import List, Tuple, Optional, Protocol
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
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
        classifier = LogisticRegression(solver='lbfgs', max_iter=200)
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
    results = fitter.predict(data2)
    if labels2 is not None:
        f1s = f1_score(results, labels2, average='macro')
        # score = fitter.score(data2, labels2)
        score = f1s
    else:
        score = 0
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
            if dt in ['float64', 'int64']:
                pass
            elif dt == 'bool':
                res[col] = res[col].astype(int)
                # print(idx, col, dt)
            else:
                res.drop(col, axis=1, inplace=True)

    if 'Label' in res.columns:
        res.drop('Label', axis=1, inplace=True)

    res.reset_index()
    return res


def check_feats(train_data: pd.DataFrame, train_labels: List[str],
                dev_data: pd.DataFrame, dev_labels: List[str],
                method: Optional[str] = None,
                minfeats: int = 1,
                maxfeats: int = 100):
    """Check performance with each of the features."""
    cols = ['Label', 'Feats']
    cols.extend(train_data.columns)
    cols.extend(['Score'])

    resdf = pd.DataFrame([], columns=cols)
    for _ in train_data.columns:
        resdf[_] = resdf[_].astype(bool)

    perms = list(product([True, False], repeat=len(train_data.columns)))
    perms2 = []
    for p in perms:
        if any(p) and np.sum(p) >= minfeats and np.sum(p) <= maxfeats:
            perms2.append(p)

    for p in tqdm(perms2):
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
            score, _probs, _results = get_fit_results(nu_train, train_labels, nu_dev, dev_labels, method)
            res['Label'] = ', '.join(rescols)  # type: ignore
            res['Feats'] = feats  # type: ignore
            res['Score'] = score  # type: ignore
            resdf = resdf.append(res, ignore_index=True)

    return resdf


def get_best_features(df: pd.DataFrame, minfeats: int = 4, topn: int = 10) -> List[List[str]]:
    """Get best features from the featuresclassifiers."""
    minfeats = df[df['Feats'] >= minfeats]
    reslist = []
    topn = minfeats.sort_values(by=['Score'], ascending=False)[:topn]  # type: ignore
    for _, row in topn.iterrows():  # type: ignore
        drop = []
        for col in topn.columns[2:-1]:  # type: ignore
            val = row[col]
            if not val:
                drop.append(col)
        # print(row['Label'], drop)
        reslist.append(drop)
    return reslist


def get_bestest_features(feats: List[List[str]],
                         traindf, testdf, testlabels,
                         embedresults, embedprobs,
                         method: Optional[str] = None,
                         autodrop: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get best features from the feature lassifiers."""
    lastbest = 0.0
    bestscore = 0.0
    bestest = Counter()  # type: ignore
    bestresult = None

    for cols in tqdm(feats):
        if autodrop:
            cols.extend(autodrop)
        zdf_tcf = get_trainable(traindf).drop(cols, axis=1)
        ddf_tcf = get_trainable(testdf).drop(cols, axis=1)
        _ddf_tcf_feat_score, ddf_tcf_feat_probs, ddf_tcf_feat_results = \
            get_fit_results(zdf_tcf, traindf['Label'], ddf_tcf, testlabels['Label'], method)
        mu_tcf = multi_results(testdf, testlabels, embedresults, embedprobs,
                               ddf_tcf_feat_results, ddf_tcf_feat_probs,
                               ['Caps', 'Hassub'], ['Quotes'])
#                               ['Caps', 'Hassub'], ['!Trans', 'Quotes'])
        co_tcf = len(mu_tcf[mu_tcf['Prediction'] == mu_tcf['Label']])
        tcf_score = co_tcf/len(mu_tcf)

        if tcf_score > bestscore:
            lastbest = bestscore  # noqa: F841
            bestscore = tcf_score
            bestresult = mu_tcf
            print("%s: %f BEST" % (', '.join(cols), tcf_score))
        elif tcf_score > bestscore - 0.001:
            print("%s: %f / %f" % (', '.join(cols), tcf_score, bestscore - tcf_score))
        bestest[', '.join(cols)] = tcf_score  # type: ignore
    return bestresult, pd.DataFrame.from_records(list(dict(bestest).items()), columns=['Columns', 'Score'])


def multi_results(df: pd.DataFrame, gold: Optional[List[str]],
                  labels1: List[str], scores1: List[float],
                  labels2: List[str], scores2: List[float],
                  lit_features: List[str] = None,
                  idiom_features: List[str] = None,
                  agreeonly: bool = False) -> pd.DataFrame:
    """Get combined results from classifiers."""
    newdf = df.copy()
    if gold is not None:
        newdf['Label'] = gold['Label']  # type: ignore
    newdf['Pred1'] = labels1
    newdf['Pred2'] = labels2
    newdf['Score1'] = scores1
    newdf['Score2'] = scores2

    agree = labels1 == labels2
    # If the predictions agree, let's just use one
    newdf.loc[agree, 'Prediction'] = newdf['Pred1']
    # Disagreements
    trans_score1 = newdf['Trans'] & (newdf['Pred1'] == '1') & (newdf['Pred2'] == '0')
    trans_score2 = newdf['Trans'] & (newdf['Pred1'] == '0') & (newdf['Pred2'] == '1')
    score1_adj = [0] * len(newdf)
    score2_adj = [0] * len(newdf)
    # Adjust score by 0.05 if Trans is True
    for idx, val in enumerate(trans_score1):
        if val:
            score1_adj[idx] = 0.05

    for idx, val in enumerate(trans_score2):
        if val:
            score2_adj[idx] = 0.05

    use_pred1 = newdf['Score1'] + score1_adj > newdf['Score2'] + score2_adj
    use_pred2 = ~use_pred1

    # print(np.sum(use_pred1), np.sum(use_pred2))
    disagree1 = ~agree & use_pred1
    disagree2 = ~agree & use_pred2
    # print(np.sum(disagree1), np.sum(disagree2))
    newdf.loc[disagree1, 'Prediction'] = newdf['Pred1']
    newdf.loc[disagree2, 'Prediction'] = newdf['Pred2']

    # Set to literal if there is disagreement and Trans is true
    # trans_ex = ~agree & newdf['Trans']
    # newdf.loc[trans_ex, 'Prediction'] = '1'

    # sub_not_trans = newdf['Hassub'] & (newdf['Trans'] == False)

    if idiom_features:
        for f in idiom_features:
            if f.startswith('!'):
                f = f[1:]
                id_mask = newdf[f] == False  # noqa: E712
            else:
                id_mask = newdf[f]
            if agreeonly:
                newdf.loc[id_mask & ~agree, 'Prediction'] = '0'
#                newdf.loc[id_mask & ~agree, 'Prediction'+f] = '0'
            else:
                newdf.loc[id_mask, 'Prediction'] = '0'
#                newdf.loc[id_mask, 'Prediction'+f] = '0'

    # print(np.sum(sub_not_trans))
    if lit_features:
        lit_mask = [False] * len(newdf)
        for f in lit_features:
            lit_mask = newdf[f]
            if agreeonly:
                newdf.loc[lit_mask & ~agree, 'Prediction'] = '1'
#                newdf.loc[lit_mask & ~agree, 'Prediction'+f] = '1'
            else:
                newdf.loc[lit_mask, 'Prediction'] = '1'
#                newdf.loc[lit_mask, 'Prediction'+f] = '1'

    # replace $ sign which messes up dataframe showing
    xy = newdf[['Previous', 'Target', 'Next']].replace({'\$': '$\$$'}, regex=True)
    newdf[['Previous', 'Target', 'Next']] = xy

    return newdf


def check_diffs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Return a dataframe where the result differ in result dataframes."""
    of = dfs[1:]
    cols = dfs[0].rename(columns={'Label': 'Label1'}).columns
    ct = 2
    for frame in of:
        nucol = 'Label'+str(ct)
        f2 = frame.rename(columns={'Label': nucol})
        ocols = f2.columns.drop(['ID', 'Language', 'Setting'])
        cols = cols.append(ocols)
        ct += 1
    print(cols)
    resdf = pd.DataFrame(columns=cols)
    for idx, row in dfs[0].iterrows():
        labels = [row['Label']]
        for other in of:
            labels.append(other.loc[idx, 'Label'])
        if len(set(labels)) == 1:
            pass
        else:
            addrow = [row['ID'], row['Language'], row['Setting']]
            for lbl in labels:
                addrow.append(lbl)
            # print(addrow)
            # print(resdf)
            resdf.loc[idx, :] = addrow
            # print(idx)

    return resdf


def get_diff_df(diffs: pd.DataFrame, frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Get best result based on dataframes."""
    resdf = frames[0]
    for idx, row in diffs.iterrows():
        numlabels = len(frames)
        labels = []
        for frame in frames:
            labels.append(frame.loc[idx, 'Label'])
        sum_1 = np.sum([int(_) for _ in labels])
        gotlabel = '1' if sum_1 > numlabels / 2 else 0
        # print(labels, sum_1, gotlabel)
        resdf.loc[idx, 'Label'] = gotlabel
    return resdf


def majority_results(df: pd.DataFrame, gold: Optional[List[str]],
                     labels1: List[str], scores1: List[float],
                     labels2: List[str], scores2: List[float],
                     labels3: List[str], scores3: List[float],
                     lit_features: List[str] = None,
                     idiom_features: List[str] = None,
                     agreeonly: bool = False) -> pd.DataFrame:
    """Majority voting classifier."""
    newdf = df.copy()
    if gold is not None:
        newdf['Label'] = gold['Label']  # type: ignore
    newdf['Pred1'] = labels1
    newdf['Pred2'] = labels2
    newdf['Pred3'] = labels3
    newdf['Score1'] = scores1
    newdf['Score2'] = scores2
    newdf['Score3'] = scores3

    idx = 0
    for l1, l2, l3 in zip(labels1, labels2, labels3):
        tot = int(l1) + int(l2) + int(l3)
        # print(idx, l1, l2, l3, tot)
        if tot > 1:
            pred = '1'
        else:
            pred = '0'

        newdf.loc[idx, 'Prediction'] = pred

        if not agreeonly or (tot == 1 or tot == 2):
            if idiom_features:
                for f in idiom_features:
                    if f.startswith('!'):
                        f = f[1:]
                        val = not newdf.loc[idx, f]
                    else:
                        val = newdf.loc[idx, f]
                    # print(idx, tot, f, val)
                    if val:
                        newdf.loc[idx, 'Prediction'] = '0'
                        newdf.loc[idx, 'Prediction'+f] = '0'

            if lit_features:
                for f in lit_features:
                    val = newdf.loc[idx, f]
                    # print(idx, tot, f, val)
                    if val:
                        newdf.loc[idx, 'Prediction'] = '1'
                        newdf.loc[idx, 'Prediction'+f] = '1'

        idx += 1
        # break
    return newdf
