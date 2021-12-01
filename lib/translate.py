"""Get backtranslations with opus-mt."""

# pylint: disable=invalid-name, line-too-long

# from typing import List
# from typing import List, Tuple, Dict
# import numpy as np
import pandas as pd
import re
from transformers import MarianMTModel, MarianTokenizer
from tqdm.notebook import tqdm
from .util import strip_accents
from .masker import get_string_diff, replace_mask_token

enpt_modelname = 'Helsinki-NLP/opus-mt-en-roa'
pten_modelname = 'Helsinki-NLP/opus-mt-roa-en'


def get_marian_models(modelname1: str = enpt_modelname, modelname2: str = pten_modelname):
    """Get marian models and tokenizers for backtranslation."""
    model1 = MarianMTModel.from_pretrained(modelname1)
    model2 = MarianMTModel.from_pretrained(modelname2)
    tokenizer1 = MarianTokenizer.from_pretrained(modelname1)
    tokenizer2 = MarianTokenizer.from_pretrained(modelname2)

    return model1, model2, tokenizer1, tokenizer2


def backtranslate(df: pd.DataFrame, model1, model2, tokenizer1, tokenizer2, batch_len: int = 100):
    """Backtranslate with model1 and model2."""
    # count = 0

    resdf = df.copy()

    en_index = df['Language'] == 'EN'
    pt_index = df['Language'] == 'PT'

    en_sentences = list(df[en_index]['Target'].values)
    # en_mwes = list(df[en_index]['MWE'].values)

    pt_sentences = list(df[pt_index]['Target'].values)
    # pt_mwes = list(df[pt_index]['MWE'].values)

    # FIXME: currently hard-coded to en/pt
    # FIXME: glg sentences?
    en_pt_sentences = []
    en_pt_en_sentences = []

    for i in tqdm(range((len(en_sentences)-1)//batch_len+1)):
        slc = en_sentences[i*batch_len:(i+1)*batch_len]
        # print(len(slc), i*batch_len, (i+1)*batch_len)
        src = ['>>por<< ' + t for t in slc]
        en_trans1 = model1.generate(**tokenizer1(src, return_tensors="pt", padding=True))
        en_trg_text = [tokenizer1.decode(t, skip_special_tokens=True) for t in en_trans1]
        # print(slc)
        # print(en_trg_text)
        en_pt_sentences.extend(en_trg_text)
        en_trans2 = model2.generate(**tokenizer2(en_trg_text, return_tensors="pt", padding=True))
        en_back_text = [tokenizer2.decode(t, skip_special_tokens=True) for t in en_trans2]
        # print(en_back_text)
        en_pt_en_sentences.extend(en_back_text)
        # break

    for idx, sent in zip(df[en_index].index, en_pt_en_sentences):
        # print(idx, sent)
        resdf.at[idx, 'BT'] = sent

    pt_en_sentences = []
    pt_en_pt_sentences = []

    for i in tqdm(range((len(pt_sentences)-1)//batch_len+1)):
        slc = pt_sentences[i*batch_len:(i+1)*batch_len]
        # print(len(slc), i*batch_len, (i+1)*batch_len)
        pt_trans1 = model2.generate(**tokenizer2(slc, return_tensors="pt", padding=True))
        pt_trg_text = [tokenizer2.decode(t, skip_special_tokens=True) for t in pt_trans1]
        # print(slc)
        # print(pt_trg_text)
        pt_en_sentences.extend(pt_trg_text)
        trg = ['>>por<< ' + t for t in pt_trg_text]
        pt_trans2 = model1.generate(**tokenizer1(trg, return_tensors="pt", padding=True))
        pt_back_text = [tokenizer1.decode(t, skip_special_tokens=True) for t in pt_trans2]
        # print(pt_back_text)
        pt_en_pt_sentences.extend(pt_back_text)
        # break

    for idx, sent in zip(df[pt_index].index, pt_en_pt_sentences):
        # print(idx, sent)
        resdf.at[idx, 'BT'] = sent

    return resdf


def record_trans(df: pd.DataFrame) -> pd.DataFrame:
    """Record translation flag."""
    resdf = df.copy()

    spacect = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):

        MWE = row['MWE']
        Target = row['Target']
        bt = row['BT']

        m = replace_mask_token(Target, MWE, '<mask>')
        tkall, lastchar = get_string_diff(Target, m, '<mask>')
        if not tkall:
            print('No match:', MWE, Target, m)

        candidates = [tkall]
        if '-' in tkall:
            candidates.append(tkall.replace('-', ' '))
        else:
            candidates.append(tkall.replace(' ', '-'))
        candidates.append(tkall.replace(' ', ''))
        tk_ = strip_accents(tkall)
        if tk_ != tkall:
            existing = list(candidates)
            for _ in existing:
                candidates.append(strip_accents(_))

        hastrans = False
        # print(1, Target)
        # print(2, bt)
        # print(3, m)
        # print(4, tkall)

        for c in candidates:
            # print(c)
            if lastchar and lastchar.isalpha():
                regex = r"(?i).*\b%s" % c.lower()
            else:
                regex = r"(?i).*\b%s\b.*" % c.lower()
            res = re.match(regex, bt)
            if not res:
                res = re.match(regex, strip_accents(bt))
            # res = re.search(r"(?i)(" + c + ")", bt)
#            if MWE == 'pé-frio':
#            if MWE == 'círculo virtuoso':
#                print(bt)
#                print(c, res)
#                print()
            if res:
                if '-' in c:
                    pass
                elif ' ' not in c:
                    spacect += 1
                hastrans = True
                break

        resdf.at[index, 'Trans'] = hastrans

    # print(spacect)
    return resdf
