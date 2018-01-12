#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:24:23 2018

@author: karolis
"""

from __future__ import unicode_literals, print_function, division
from torch.autograd import Variable
import numpy as np
import unicodedata
import string
import torch
import glob


LETTERS = {l: idx for idx, l in enumerate(string.ascii_letters + " .,;'")}


def unicode_to_ascii(s):
    '''
        See https://goo.gl/NnybE1
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        # The character category "Mn" stands for Nonspacing_Mark
        if unicodedata.category(c) != 'Mn'
    )


def get(data_path):
    def read(file):
        names = open(
            file,
            encoding='utf-8',
        ).read().strip().split('\n')
        return [unicode_to_ascii(name) for name in names]

    categories, names = [], {}

    for file in glob.glob('/'.join([data_path, '*.txt'])):
        category = file.split('/')[-1].split('.')[0]
        categories.append(category)
        names[category] = read(file)

    return categories, names


def letter_to_tensor(letter):
    '''
        Returns a one-hot encoded 1 x len(letters) tensor
    '''
    tensor = torch.zeros(1, len(LETTERS))
    tensor[0][LETTERS[letter]] = 1
    return tensor


def name_to_tensor(name):
    '''
        Returns an array of one-hot tensors for a name,
        i.e. len(name) x 1 x len(letters)
    '''
    tensor = torch.zeros(len(name), 1, len(LETTERS))

    for idx, l in enumerate(name):
        tensor[idx][0][LETTERS[l]] = 1

    return tensor


def random_example(categories, names):
    category = np.random.choice(categories)
    name = np.random.choice(names[category])
    category_tensor = Variable(
        torch.LongTensor([
            categories.index(category)
        ])
    )
    name_tensor = name_to_tensor(name)

    return category, name, category_tensor, name_tensor
