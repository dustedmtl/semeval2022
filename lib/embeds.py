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
