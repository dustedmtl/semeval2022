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
    """Save picture to disk, append date."""
    datestr = get_datestr()
    fmt = "%s/%s_%s_%d.%s"
    filename = fmt % (path, name, datestr, ext, imgfmt)
    print('Saving image to', filename)
    pic.savefig(filename, bbox_inches='tight')


def add_hlines(latex: str) -> str:
    """Add hlines to latex table."""
    lines = []
    lineno = 0
    for line in latex.split('\n'):
        lineno += 1
        if lineno > 2 and 'tabular' in line:
            lines.append('\hline')
        lines.append(line)
        if lineno in [1,2]:
            lines.append('\hline')
    return '\n'.join(lines)


def save_table(df: pd.DataFrame,
               name: str, path: str = 'paper/tables',
               ext=1, index: bool = True,
               colformat: Dict = {}):
    """Save dataframe to disk, append date."""
    datestr = get_datestr()
    fmt = "%s/%s_%s_%d.tex"
    filename = fmt % (path, name, datestr, ext)
    ldf = df.copy()
    for rmcol in ['C', 'C2', 'DataID']:
        if rmcol in ldf.columns:
            ldf = ldf.drop(rmcol, axis=1)
    for col, t in zip(ldf.columns, ldf.dtypes):
        if t == bool:
            ldf = ldf.astype({col: str})
    latex = ldf.style.format(precision=3).format_index("\\textbf{{{}}}", escape="latex", axis=1)
    # Remove index column
    if not index:
        latex = latex.hide()
    # Apply column formatting overrides
    colfmt = ''
    for col, dt in zip(ldf.columns, ldf.dtypes):
        if col in colformat:
            colfmt += colformat[col]
        elif dt in [object, bool]:
            colfmt += 'l'
        else:
            colfmt += 'r'
    latex = latex.to_latex(column_format=colfmt).replace('_', '\\_')
    # print(latex)
    latex = add_hlines(latex)
    # print(latex)
    print('Saving dataframe to', filename)
    with open(filename, 'w') as f:
        f.write(latex)
