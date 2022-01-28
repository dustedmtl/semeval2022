"""Data visualization."""

# pylint: disable=invalid-name, line-too-long, unused-variable, unnecessary-comprehension

# from typing import List, Tuple, Optional, Protocol
import numpy as np
import pandas as pd
# import scipy as sp
import seaborn as sns
import matplotlib as plt
from sklearn.metrics import confusion_matrix


def df_heatmap(df: pd.DataFrame, gold: pd.DataFrame, col='Label') -> plt.axes:
    """Get counts and labels as a heatmap."""
    ylabels = df[col].unique()
    if col != 'Label':
        cf = confusion_matrix(df[col].map({ylabels[0]: '0', ylabels[1]: '1'}), gold['Label'])
    else:
        cf = confusion_matrix(df['Label'], gold['Label'])
    counts = [v for v in cf.flatten()]
    pct = [v for v in cf.flatten()/np.sum(cf)]
    labels = ["%s\n%.1f%%" % (v1, 100*v2) for v1, v2 in zip(counts, pct)]
    labels = np.asarray(labels).reshape(2, 2)
    if col != 'Label':
        ax = sns.heatmap(cf, annot=labels, cmap='Blues', fmt='', yticklabels=ylabels)
    else:
        ax = sns.heatmap(cf, annot=labels, cmap='Blues', fmt='')
    return ax
