#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 00:29:39 2017

@author: karolis
"""

import re
from collections import Counter


LETTERS = 'abcdefghijklomnpqrstuvwxyz'


WORDS = Counter(
    re.findall(
        r'\w+', 
        open(
            'big.txt'
        ).read().lower()
    )
)
    
    
def P(word, N=sum(WORDS.values())):
    return WORDS[word] / N


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [left + right[1:] for left, right in splits if right]
    transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
    replaces = [left + letter + right[1:] for left, right in splits if right for letter in LETTERS]
    inserts = [left + letter + right for left, right in splits for letter in LETTERS]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    return (edit2 for edit1 in edits1(word) for edit2 in edits1(edit1))


def known(words):
    return set(word for word in words if word in WORDS)


def spell_check(word):
    '''
        Returns 3 most likely corrections
    '''
    candidates = (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
    return sorted(candidates, key=P, reverse=True)[:3]
