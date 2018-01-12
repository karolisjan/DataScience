#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:47:48 2017

@author: karolis
"""

import numpy as np


class RNN:

    def __init__(
        self,
        hidden_layer_size=100,
        bptt_truncate=4,
    ):

        self.hidden_layer_size = hidden_layer_size
        self.bptt_truncate = bptt_truncate

    def __forward_propagation(self, X):
        pass

    def fit(self, X, y):
        '''
            Weights are initialised according to:

            Glorot, X. and Bengio, Y., 2010, March.
            Understanding the difficulty of training
            deep feedforward neural networks.
            In Proceedings of the Thirteenth International Conference
            on Artificial Intelligence and Statistics (pp. 249-256).

            Parameters U, V, W are shared across all time steps t:

            s_t = f(Ux_t + Ws_t-1) where f is an activation function, e.g. tanh
            o_t = softmax(Vs_t)
        '''

        n_samples, n_features = X.shape

        self.U = np.random.uniform(
            -np.sqrt(1.0 / n_features),
            np.sqrt(1.0 / n_features),
            (self.hidden_layer_size, n_features)
        )

        self.V = np.random.uniform(
            -np.sqrt(1.0 / self.hidden_layer_size),
            np.sqrt(1.0 / self.hidden_layer_size),
            (n_features, self.hidden_layer_size)
        )

        self.W = np.random.uniform(
            -np.sqrt(1.0 / self.hidden_layer_size),
            np.sqrt(1.0 / self.hidden_layer_size),
            (self.hidden_layer_size, self.hidden_layer_size)
        )
