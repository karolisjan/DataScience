l#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 07:02:50 2017

@author: karolis
"""

import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


LEMMATIZER = WordNetLemmatizer()
BOOK_TITLES = [l.rstrip() for l in open('all_book_titles.txt')]
STOPWORDS = set(w.rstrip() for w in open('stopwords.txt'))
STOPWORDS = STOPWORDS.union({
    'introduction', 'edition', 'series',
    'application', 'approach', 'card', 'access',
    'package', 'plus', 'etext', 'brief', 'vol',
    'fundamental', 'guid', 'essential', 'printed',
    'third', 'second', 'fourth'
})


def tokenize(text):
    tokens = [
        token
        for token in nltk.tokenize.word_tokenize(text.lower())
        if len(token) > 2
    ]
    return [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOPWORDS and not any(c.isdigit() for c in token)
    ]


word_index_map = {}
current_index = 0
all_tokens = []
titles = []
index_word_map = []

for title in BOOK_TITLES:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        titles.append(title)
        all_tokens.append(tokenize(title))
        for token in all_tokens[-1]:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except BaseException as e:
        print("WARNING", e)
        pass


def tokens_to_vector(tokens):
    vector = np.zeros(len(word_index_map))
    for token in tokens:
        idx = word_index_map[token]
        vector[idx] = 1
    return vector


N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))

for i, tokens in enumerate(all_tokens):
    X[:, i] = tokens_to_vector(tokens)

svd = TruncatedSVD()
Z = svd.fit_transform(X)

plt.scatter(Z[:, 0], Z[:, 1])
for i in range(D):
    plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))
plt.show()
