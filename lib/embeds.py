"""Get embeddings with sentence transformers.."""

# pylint: disable=invalid-name, line-too-long

from typing import List
# from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def get_embeddings(df: pd.DataFrame,
                   modelname: str = 'all-MiniLM-L6-v2',
                   append: List[str] = None):
    """Get sentence embeddings with the given model."""
    model = SentenceTransformer(modelname)
    sentences = df['Target']
    if append:
        print('Appending to Target: %s' % ', '.join(append))
        for a in append:
            sentences = sentences + df[a]

    print('Average sentence length: %.2f' % np.average([len(_) for _ in sentences]))
    embeddings = model.encode(sentences)
    return embeddings


def get_cosine_diffs(emb1, emb2) -> np.ndarray:
    """Get cosine different."""
    diff = np.array([])

    for e1, e2 in zip(emb1, emb2):
        d = np.dot(e1, e2.T)/(np.linalg.norm(e1)*np.linalg.norm(e2))
        diff = np.append(diff, d)

    return diff


def get_prev_next_diff(df: pd.DataFrame, modelname) -> pd.DataFrame:
    """Get previous/next diffs."""
    model = SentenceTransformer(modelname)
    prev_embeddings = model.encode(df['Previous'])
    next_embeddings = model.encode(df['Next'])
    target_embeddings = model.encode(df['Target'])
    mwe_embeddings = model.encode(df['MWE'])

    resdf = df.copy()
    resdf['Prevdiff'] = get_cosine_diffs(prev_embeddings, target_embeddings)
    resdf['Nextdiff'] = get_cosine_diffs(next_embeddings, target_embeddings)
    resdf['MWEdiff'] = get_cosine_diffs(mwe_embeddings, target_embeddings)

    return resdf
