#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 04:40:53 2017

@author: karolis
"""

import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


WORDNET_LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(w.rstrip() for w in open('stopwords.txt'))


def tokenize(text):
    tokens = [
        token
        for token in nltk.tokenize.word_tokenize(text.lower())
        if len(token) > 2
    ]
    return [
        WORDNET_LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOPWORDS
    ]


positive_reviews = BeautifulSoup(
    open('sorted_data_acl/electronics/positive.review'),
    'lxml'
)
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(
    open('sorted_data_acl/electronics/negative.review'),
    'lxml'
)
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

word_index_map = {}
current_index = 0

positive_tokens, negative_tokens = [], []

for review in positive_reviews:
    tokens = tokenize(review.text)
    positive_tokens.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = tokenize(review.text)
    negative_tokens.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


def tokens_to_vector(tokens, label):
    vector = np.zeros(len(word_index_map) + 1)
    for token in tokens:
        idx = word_index_map[token]
        vector[idx] += 1
    vector = vector / vector.sum()
    vector[-1] = label
    return vector


N = len(positive_tokens) + len(negative_tokens)
data = np.zeros((N, len(word_index_map) + 1))

idx = 0

for tokens in positive_tokens:
    data[idx, :] = tokens_to_vector(tokens, label=1)
    idx += 1

for tokens in negative_tokens:
    data[idx, :] = tokens_to_vector(tokens, label=0)
    idx += 1

np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1]

X_train = X[:-100, ]
y_train = y[:-100, ]
X_test = X[-100:, ]
y_test = y[-100:, ]

model = LogisticRegression()
model.fit(X_train, y_train)

print("Classification rate", model.score(X_test, y_test))

# Prints out the assocaited weight with the word
threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)
