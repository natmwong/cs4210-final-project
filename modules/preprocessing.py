"""
Preprocessing module for Steam Review Sentiment Analysis.
Contains functions for text preprocessing, including contraction expansion
and negation marking.
"""

import re
from nltk.tokenize import word_tokenize
import numpy as np
from .stopwords import NLTK_STOPWORDS, STOPWORDS, NEGATION_TRIGGERS

def _expand_contractions(text: str) -> str:
    """Expand negation contractions before punctuation stripping."""
    # Replace common contractions with their expanded forms.
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot",   text)
    text = re.sub(r"n't",   " not",     text)
    return text

def _apply_negation(tokens: list) -> list:
    """
    Replace <trigger> <word> pairs with a single NOT_<word> token.
    E.g. ['not', 'good'] → ['NOT_good']
    """
    # Iterate through tokens and apply negation marking.
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in NEGATION_TRIGGERS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in NEGATION_TRIGGERS:
                result.append(f"NOT_{nxt}")
                i += 2
                continue
            i += 1
            continue
        result.append(tokens[i])
        i += 1
    return result

def preprocess_text(text: str) -> str:
    """Preprocess text by expanding contractions, marking negations, and removing stopwords."""
    # Convert text to lowercase, expand contractions, and tokenize.
    text = str(text).lower()
    text = _expand_contractions(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = _apply_negation(tokens)
    # Filter out stopwords and short tokens.
    tokens = [
        t for t in tokens
        if (t.startswith('NOT_') or (t not in NLTK_STOPWORDS and t not in STOPWORDS))
        and len(t) > 2
    ]
    return ' '.join(tokens)

def review_keywords(review_text: str, model, vectorizer, n: int = 5):
    """Extract keywords and sentiment from a review using the model and vectorizer."""
    # Preprocess the review text and vectorize it.
    clean      = preprocess_text(review_text)
    vec        = vectorizer.transform([clean])
    proba      = model.predict_proba(vec)[0]
    pred       = int(np.argmax(proba))
    confidence = proba[pred]

    # Extract top keywords based on model coefficients.
    names = np.array(vectorizer.get_feature_names_out())
    coef  = model.coef_[0]
    rv    = vec.toarray()[0]
    nz    = np.where(rv > 0)[0]

    if len(nz):
        scores   = coef[nz] * rv[nz]
        top      = np.argsort(np.abs(scores))[-n:][::-1]
        keywords = [(names[nz[i]], scores[i]) for i in top]
    else:
        keywords = []

    # Determine sentiment based on prediction.
    sentiment = 'Positive' if pred == 1 else 'Negative'
    return sentiment, confidence, keywords
