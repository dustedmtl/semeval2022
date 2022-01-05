"""Mask-filling."""

# pylint: disable=invalid-name, line-too-long, too-many-locals

# from typing import List, Tuple, Dict
from typing import Optional
import re
# import numpy as np
import pandas as pd
from transformers import pipeline
from .util import strip_accents
from tqdm.notebook import tqdm


def replacer2(string: str, mwe: str, mask_token: str, splitter: str) -> str:
    """
    Find the MWE in the string and replace it with the mask token.

    Handles: inflection, accents and different separators.
    """
    parts = mwe.split(splitter)
    u_mwe = strip_accents(mwe)
    accented = u_mwe != mwe

    if splitter == ' ':
        splitter2 = '-'
    else:
        splitter2 = ' '

    # For each iteration:
    # - create regex (if applicable)
    # - create alternate splitter variant
    # - if accented
    #   - create accented variant
    #   - create accented splitter variant

    for i in range(5):
        nuparts = []
        u_nuparts = []

        for p in parts:
            # p2 = re.sub(r'\w(\.+)?$', "\\\w+", p)
            if i == 0:
                nuparts.append(p)
                u_nuparts.append(strip_accents(p))
                continue
            # print(p[:-i-1])
            p2 = p[:-i-1] + "\w+"
            nuparts.append(p2)
            # print(p2)
            # print(strip_accents(p2))
            u_nuparts.append(strip_accents(p2))

        # print(nuparts)
        # print(u_nuparts)
        subber = r'(?i)' + splitter.join([ '(' + w + ')' for w in nuparts])

        # print(subber, 'to', mask_token)
        m = re.sub(subber, mask_token, string, 1)
        if m != string:
            # print('Got', m)
            break
        else:
            alt_subber = r'(?i)' + splitter2.join([ '(' + w + ')' for w in nuparts])
            alt_mask_token = mask_token.replace(splitter, splitter2)
            # print(alt_subber, 'to', alt_mask_token)
            m = re.sub(alt_subber, alt_mask_token, string, 1)
            if m != string:
                break
            elif accented:
                u_subber = r'(?i)' + splitter.join([ '(' + w + ')' for w in u_nuparts])
                # print(u_subber, 'to', mask_token)
                m = re.sub(u_subber, mask_token, string, 1)
                if m != string:
                    # print('Got', m)
                    break

    return m


def get_new_mask_token(mask: str, splitter: str, mask_token: str) -> str:
    """Get new mask token iteratively with regex."""
    # preserve replaced part inflection
    mask_parts = mask.split(splitter)
    # print(mask, mask_parts)
    nu_mask_parts = []
    for idx, token in enumerate(mask_parts):
        if token == mask_token:
            nu_mask_parts.append(token)
        else:
            nu_mask_parts.append('\\' + str(idx+1))
    nu_mask_token = splitter.join(nu_mask_parts)
    return nu_mask_token


def replace_mask_token(string: str, mwe: str, mask_token: str, orig_mask: Optional[str] = None):
    """Replace mask token in string."""
    # print(string, mwe, mask_token)
    torep = '(?i)' + mwe
    m = re.sub(torep, mask_token, string, 1)

    # Couldn't find a replacement, let's try something new
    if m == string:
        splitter = ' '
        if '-' in mwe:
            splitter = '-'

        if orig_mask and splitter in mask_token:
            mask_token = get_new_mask_token(mask_token, splitter, orig_mask)
            # print(mask_token)

        m = replacer2(string, mwe, mask_token, splitter)

    return m


def get_masked_tokens(df: pd.DataFrame, model, limit_term: Optional[str] = None) -> pd.DataFrame:
    """Get masked token for the dataframe."""
    res = df.copy()

    mwe_idx = list(df.columns).index('MWE')
    target_idx = list(df.columns).index('Target')

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        target = row[target_idx]
        mwe = row[mwe_idx]

        if limit_term:
            if limit_term not in mwe:
                continue
        mask_token = model.tokenizer.mask_token

        # Substitution (whole term)
        m = replace_mask_token(target, mwe, mask_token)
        tkall, _ = get_string_diff(target, m, mask_token)

        # Get top 5 terms
        ks = model(m, top_k=5)

        terms = []
        has_component = False
        shorts = 0

        # Go through the top terms
        # Strings shorter than 3 characters are ignored
        # So are non-words

        found_term_index = None
        found_term_score = None

        for d_idx, d in enumerate(ks):
            term = d['token_str']
            terms.append(term)
            if len(term) < 3:
                shorts += 1
            else:
                if re.match(r"\W", term):
                    # Discard terms that have non-word characters
                    continue
                # Result term must match the MWE part exactly
                regex = r"(?i).*\b%s\b.*" % term.lower()
                # print(mwe,regex,term)
                gotit = False
                if re.match(regex, mwe):
                    gotit = True
                elif re.match(regex, tkall):
                    # print('Got', term, 'in found:', tkall)
                    gotit = True
                if gotit:
                    has_component = True
                    if not found_term_index:
                        found_term_index = d_idx + 1
                    if not found_term_score:
                        found_term_score = d['score']

        if not found_term_index:
            found_term_index = 10
            found_term_score = -1

        top_score = ks[0]['score']
        # print(target, terms, top_score)

        res.at[idx, 'Top terms'] = ', '.join(terms)
        res.at[idx, 'Top score'] = top_score
        res.at[idx, 'Hassub'] = has_component
        res.at[idx, 'Short'] = shorts

        res.at[idx, 'FoundIdx'] = found_term_index
        res.at[idx, 'FoundScore'] = found_term_score

        if '-' in mwe:
            parts = mwe.split('-')
        else:
            parts = mwe.split()

        # Replace one part or the other
        if len(parts) < 2:
            continue

        # Replace first part of the MWE
        first_terms = []
        first_top_score = 0
        first_shorts = 0

        firstrep = mask_token + ' ' + parts[1]
        # mf = re.sub(torep, firstrep, target, 1)
        mf = replace_mask_token(target, mwe, firstrep, mask_token)
        # print(firstrep, mf)

        ksf = model(mf, top_k=5)
        for d in ksf:
            term = d['token_str']
            if term.lower() == parts[0].lower():
                # print('OI! Got', parts[0], '=', term)
                continue
            first_terms.append(term)
            if len(term) < 3:
                first_shorts += 1
                continue
            if first_top_score == 0:
                first_top_score = d['score']

        # print(first_terms, first_top_score)

        # Replace second part of the MWE
        second_terms = []
        second_top_score = 0
        second_shorts = 0

        secondrep = parts[0] + ' ' + mask_token
        # ms = re.sub(torep, secondrep, target, 1)
        ms = replace_mask_token(target, mwe, secondrep, mask_token)
        # print(secondrep, ms)

        kss = model(ms, top_k=5)
        for d in kss:
            term = d['token_str']
            # print(term.lower(), parts[1].lower())
            if term.lower() == parts[1].lower():
                # print('OI! Got', parts[1], '=', term)
                continue
            second_terms.append(term)
            if len(term) < 3:
                second_shorts += 1
                continue
            if second_top_score == 0:
                second_top_score = d['score']

        # print(second_terms, second_top_score)

        res.at[idx, 'Top terms 1'] = ', '.join(first_terms)
        res.at[idx, 'Top score 1'] = first_top_score
        res.at[idx, 'Top terms 2'] = ', '.join(second_terms)
        res.at[idx, 'Top score 2'] = second_top_score
        res.at[idx, 'FS'] = first_shorts
        res.at[idx, 'SS'] = second_shorts

    return res


def get_masked_features(df: pd.DataFrame, modelname: str = "xlm-roberta-base") -> pd.DataFrame:
    """Get masked features with the selected model."""
    unmasker = pipeline("fill-mask", model=modelname)
    resdf = get_masked_tokens(df, unmasker)
    return resdf


def get_string_diff(string, m, mask_token):
    """Get string difference based on mask token."""
    # _, s2 = m.split(mask_token)
    # second = string.index(s2)

    maskpos = m.index(mask_token)
    maskendpos = maskpos + len(mask_token)

    if maskendpos == len(m):
        # mask at the end of the string
        secpos = None
        secpart = ""
    else:
        # the ending string
        secpart = m[maskendpos:]
        # this is where the ending starts in the original string
        secpos = string.index(secpart, maskpos)

    # print(m, maskendpos, len(m), secpos, secpart)
    mwe = string[maskpos:secpos]
#    mwe = string[maskpos:second]
#    if _mwe != mwe:
#        print("Couldn't get %s from: %s" % (mask_token, m))
#    print(mwe, _mwe)
    lastchar = None
    if secpos:
        lastchar = string[secpos]
    # print(mwe, lastchar)
    return mwe, lastchar
