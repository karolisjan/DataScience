#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 07:54:35 2017

@author: karolis
"""

import nltk
import random
import numpy as np
from bs4 import BeautifulSoup


positive_reviews = BeautifulSoup(
    open('sorted_data_acl/electronics/positive.review'),
    'lxml'
)
positive_reviews = positive_reviews.findAll('review_text')

trigrams = {}
for review in positive_reviews:
    tokens = nltk.tokenize.word_tokenize(review.text.lower())
    for i in range(len(tokens) - 2):
        key = (tokens[i], tokens[i + 2])
        if key not in trigrams:
            trigrams[key] = []
        trigrams[key].append(tokens[i + 1])

trigram_probabilities = {}
for key, words in trigrams.items():
    if len(set(words)) > 1:
        d = {}
        for word in words:
            if word not in d:
                d[word] = 0
            d[word] += 1
        for word, count in d.items():
            d[word] = float(count) / len(words)
        trigram_probabilities[key] = d


def random_sample(d):
    r = np.random.rand()
    cumulative = 0
    for word, prob in d.items():
        cumulative += prob
        if r < cumulative:
            return word


def test(p_replace=0.2):
    review = random.choice(positive_reviews)
    review = review.text.lower()
    print("ORIGINAL:", review)
    tokens = nltk.tokenize.word_tokenize(review)
    for i in range(len(tokens) - 2):
        if np.random.rand() < p_replace:
            key = (tokens[i], tokens[i + 2])
            if key in trigram_probabilities:
                word = random_sample(trigram_probabilities[key])
                tokens[i + 1] = word
    print("SPUN:\n", ' '.join(tokens))
