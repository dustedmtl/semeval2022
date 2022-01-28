"""Utilities for semeval 2022 dataframes."""

# pylint: disable=invalid-name

import os
import csv
from typing import List, Tuple, Dict, Optional
import unicodedata
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: str, delimiter: str = ',') -> Tuple[Optional[List], List]:
    """CSV load function from SemEval2022."""
    header = None
    data: List[str] = []
    with open(path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            if header is None:
                header = row
                continue
            data.append(row)
    return header, data


def load_df(path: str, delimiter: str = ',') -> pd.DataFrame:
    """Return Dataframe from CSV load."""
    header, data = load_csv(path, delimiter=delimiter)
    df = pd.DataFrame(data, columns=header)
    return df


def load_csv_dataframes(path: str) -> Dict[str, pd.DataFrame]:
    """Load dataframes from a single directory."""
    frames = dict()
    files = os.listdir(path)
    for file in files:
        df = load_df(os.path.join(path, file))
        frames[file] = df
    return frames


def get_counts(dataframe):
    """Get counts from a dataframe."""
    df_group = dataframe.groupby(['Language', 'MWE', 'Label'],
                                 as_index=False)['ID'].count().rename(columns={'ID': 'count'}).sort_values('count')
    df_counts = pd.DataFrame(columns=['Language', 'MWE', '0 (Idiomatic)', '1 (Literal)'])

    for _index, row in df_group.iterrows():
        Language = row['Language']
        MWE = row['MWE']
        Label = row['Label']
        count = row['count']

        if Label == '0':
            Label = '0 (Idiomatic)'
        else:
            Label = '1 (Literal)'

        target = df_counts[(df_counts['Language'] == Language) & (df_counts['MWE'] == MWE)]
        if target.empty:
            # print(target)
            df_counts = df_counts.append({'Language': Language, 'MWE': MWE, Label: count}, ignore_index=True)
        else:
            # print(target)
            df_counts.loc[(df_counts['Language'] == Language) & (df_counts['MWE'] == MWE), Label] = count

    df_counts.fillna(0, inplace=True)
    df_counts['Total'] = df_counts['0 (Idiomatic)'] + df_counts['1 (Literal)']
    df_counts['Pct literal'] = df_counts['1 (Literal)'] / df_counts['Total']

    for row in dataframe.groupby(['Language', 'MWE']).mean().iterrows():
        idx, pct_correct = row
        lang, mwe = idx
        # print(lang, mwe, pct_correct.values[0])
        df_counts.loc[(df_counts['Language'] == lang) &
                      (df_counts['MWE'] == mwe), 'Pct correct'] = pct_correct.values[0]

    return df_counts


def strip_accents(text: str) -> str:
    """Strip accents from text string."""
    text = unicodedata.normalize('NFD', text)
    # textb is bytes
    textb = text.encode('ascii', 'ignore')
    text = textb.decode("utf-8")
    return text


def save_pickle(df: pd.DataFrame, basename: str, ext: int = 1):
    """Save dataframe to disk, append date."""
    datestr = get_datestr()
    fmt = "%s_%s_%d.pkl"
    filename = fmt % (basename, datestr, ext)
    df.to_pickle(filename)
    print('Saved dataframe to', filename)


def get_datestr():
    """Get current date string."""
    currdate = datetime.now()
    datestr = currdate.strftime("%Y%m%d")
    return datestr


def save_picture(pic: plt.figure, name: str, path: str = 'paper/figures',
                 imgfmt: str = 'png', ext=1):
    """Save dataframe to disk, append date."""
    datestr = get_datestr()
    fmt = "%s/%s_%s_%d.%s"
    filename = fmt % (path, name, datestr, ext, imgfmt)
    print(filename)
    pic.savefig(filename, bbox_inches='tight')
