"""Get sentiment."""

# pylint: disable=invalid-name, line-too-long, too-many-locals

# from typing import List, Tuple, Dict
# from typing import Optional
# import re
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def get_classifier_tokenizer(modelname: str = MODEL):
    """Get sentiment classifier and tokenizer."""
    sentiment_tokenizer = AutoTokenizer.from_pretrained(modelname)
    config = AutoConfig.from_pretrained(modelname)

    sentiment_classifier = AutoModelForSequenceClassification.from_pretrained(modelname)
    sentiment_classifier.save_pretrained('xlm-sentiment')
    sentiment_tokenizer.save_pretrained('xlm-sentiment')

    return sentiment_classifier, sentiment_tokenizer, config


def preprocess(text: str) -> str:
    """Preprocess text."""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def get_sentiment(text: str, model, tokenizer, config) -> float:
    """Get neutral sentiment for the text."""
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    neutral = 0
    # Print labels and scores
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        if label == 'Neutral':
            neutral = s
        # print(f"{i+1}) {l} {np.round(float(s), 4)}")
    return neutral


def get_df_sentiments(df: pd.DataFrame, model, tokenizer, config, context=False):
    """Get sentence embeddings with the given model."""
    res = df.copy()
    print(df.columns)

    # mwe_idx = list(df.columns).index('MWE')
    target_idx = list(df.columns).index('Target')
    prev_idx = list(df.columns).index('Previous')
    next_idx = list(df.columns).index('Next')

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        previous = row[prev_idx]
        target = row[target_idx]
        nxt = row[next_idx]
        # mwe = row[mwe_idx]

        text = target

        if context:
            text = ' '.join([previous, target, nxt])

        score = get_sentiment(text, model, tokenizer, config)
        res.at[idx, 'Sentiment'] = score

    return res
