#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:52:53 2017

@author: karolis
"""

import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder


class NeuralNetwork:

    def __init__(
            self,
            hidden_layer_sizes=(100,),
            learning_rate=1e-3,
            regularization_rate=1e-4,
            epochs=1000,
            verbose=False
    ):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.epochs = epochs
        self.verbose = verbose

    def __make_model(self, layer_sizes):

        self.W, self.b = [], []

        for i in range(len(layer_sizes) - 1):
            self.W.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            )
            self.b.append(
                np.zeros((1, layer_sizes[i + 1]))
            )

    def __forward_propagation(self, X):

        Z = X.dot(self.W[0]) + self.b[0]
        self.A = []

        for i in range(1, len(self.W)):
            A = np.tanh(Z)
            self.A.append(A)
            Z = A.dot(self.W[i]) + self.b[i]

        y_hat = np.exp(Z)

        # Softmax
        return y_hat / np.sum(y_hat, axis=1, keepdims=True)

    def __backpropagation(self, X, y, y_hat):

        delta = y_hat - y
        A = [X] + self.A
        self.dW = [None] * len(A)
        self.db = [None] * len(A)

        for i in range(len(A) - 1, -1, -1):
            self.dW[i] = (A[i].T).dot(delta)
            self.db[i] = np.sum(delta, axis=0)
            delta = np.multiply(
                (1 - np.power(A[i], 2)),
                delta.dot(self.W[i].T)
            )

    def __update_model(self):

        for i in range(len(self.W)):
            self.dW[i] += self.regularization_rate * self.W[i]
            self.W[i] += -self.learning_rate * self.dW[i]
            self.b[i] += -self.learning_rate * self.db[i]

    def score(self, X, y):

        y_hat = self.__forward_propagation(X)
        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).todense()

        return log_loss(
            np.argmax(y_hat, axis=1),
            np.argmax(y, axis=1)
        )

    def fit(self, X, y):

        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).todense()

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        layer_sizes = (n_features,) + self.hidden_layer_sizes + (n_outputs,)

        self.__make_model(layer_sizes)

        for epoch in range(self.epochs):
            y_hat = self.__forward_propagation(X)
            self.__backpropagation(X, y, y_hat)
            self.__update_model()

            if self.verbose:
                print("Epoch: %d, loss: %.4f"
                      % (epoch + 1, self.score(X, y)))

        return self

    def predict(self, X):

        y_hat = self.__forward_propagation(X)
        return np.argmax(y_hat, axis=1)
