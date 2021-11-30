"""Get binary features."""

# pylint: disable=invalid-name, line-too-long, too-many-locals

# from typing import List, Tuple, Dict
# from typing import Optional
import re
# import numpy as np
import pandas as pd


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get binary features from the dataframe.

    Features: Quotes, Caps.
    """
    results = df.copy()

    print(df.columns)
    mwe_idx = list(df.columns).index('MWE')
    target_idx = list(df.columns).index('Target')

    for idx, row in df.iterrows():
        target = row[target_idx]
        mwe = row[mwe_idx]

        # Quotes: more likely to be idiomatic
        quoted = False
        if "\"" in target:
            # print(target)
            res = re.search(r'(?i)(\"' + mwe + '\")', target)
            if res:
                quoted = True
                # print(target)
                # print(res.groups())

        # Capitalization: more likely (certain?) to be literal
        # FIXME: Check if most words in target are capitalized: Could be a new headline
        caps = False
        mwecaps = ' '.join([w.capitalize() for w in mwe.split()])
        # print(mwecaps, target)
        res2 = re.search(r'(' + mwecaps + ')', target)
        # print(res2)
        if res2:
            caps = True
            # print(target)
            # print(res2.groups())

        results.at[idx, 'Quotes'] = quoted
        results.at[idx, 'Caps'] = caps

        results['Quotes'] = results['Quotes'].astype(bool)
        results['Caps'] = results['Caps'].astype(bool)

    return results
