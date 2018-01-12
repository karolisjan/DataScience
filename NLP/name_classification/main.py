#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:05:28 2018

@author: karolis
"""

import utils
import numpy as np
from rnn import RNN


if __name__ == "__main__":

    categories, names = utils.get('data/names')

    n_input = len(utils.LETTERS)
    n_hidden = 128
    n_output = len(categories)

    rnn = RNN(n_input, n_hidden, n_output)
    rnn.train(
        categories,
        names,
        learning_rate=0.005,
        epochs=100,
        verbose=True
    )