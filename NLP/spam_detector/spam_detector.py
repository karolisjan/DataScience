#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 03:56:58 2017

@author: karolis
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('spambase/spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, :48]
y = data[:, -1]

X_train = X[:-100,]
y_train = y[:-100,]

X_test = X[-100:,]
y_test = y[-100:,]

model = MultinomialNB()
model.fit(X_train, y_train)

print("Classification rate for NB:", model.score(X_test, y_test))