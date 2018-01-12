#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:49:23 2018

@author: karolis
"""

import utils
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable


class RNN(nn.Module):
    '''
        See https://i.imgur.com/Z2xbySO.png
    '''

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.cost_function = nn.NLLLoss()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def __update(
    self,
    learning_rate,
    category_tensor,
    name_tensor
):

        hidden = self.init_hidden()
        self.zero_grad()

        for i in range(name_tensor.size()[0]):
            output, hidden = self(name_tensor[i], hidden)

        loss = self.cost_function(output, category_tensor)
        loss.backward()

        for par in self.parameters():
            par.data.add_(-learning_rate, par.grad.data)

        return output, loss.data[0]

    def translate_output(self, output, categories):
        top_n, top_idx = output.data.topk(1)
        category_idx = top_idx[0][0]
        return categories[category_idx], category_idx

    def train(
        self,
        categories,
        names,
        learning_rate,
        epochs,
        verbose=False
    ):

        if verbose:
            pbar = tqdm(total=epochs)

        history = []

        for epoch in range(epochs):
            category, name, category_tensor, name_tensor = \
                utils.random_example(categories, names)
            output, loss = self.__update(
                learning_rate,
                category_tensor,
                name_tensor
            )
            history.append(loss)

            if verbose:
                pbar.update(epoch)
                tqdm.write("Loss: %.4f" % loss)

        return history
