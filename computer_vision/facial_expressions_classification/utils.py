# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:28:47 2017

@author: Karolis
"""
import itertools
import warnings
import numpy as np
from copy import deepcopy
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def k_fold_validation(clf, X, y, cv):
    warnings.filterwarnings("ignore", category=DeprecationWarning)    
    
    training_scores, test_scores = [], []
    best_score = float("inf")
    best_clf = None
    
    k_folds = KFold(n_splits=cv)
            
    for training_idx, test_idx in k_folds.split(X):                
        X_train, y_train = X[training_idx], y[training_idx]
        X_test, y_test = X[test_idx], y[test_idx]
       
        clf.fit(X_train, y_train) 
        
        training_scores.append(clf.score(X_train, y_train))
        score = clf.score(X_test, y_test)
        test_scores.append(score)
        
        if score < best_score:
            best_clf = deepcopy(clf)

    return np.array(training_scores), np.array(test_scores), best_clf

    
def plot_confusion_matrix(y_true, y_hat,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    confusion_matrix = metrics.confusion_matrix(y_true, y_hat)
    classes = np.unique(y_true)
    
    fig = plt.figure(figsize=(8, 6), dpi=90)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        confusion_matrix = np.round((confusion_matrix.astype('float') 
                                     / confusion_matrix.sum(axis=1)[:, np.newaxis]), 2)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), 
                                  range(confusion_matrix.shape[1])):
        
        plt.text(j, 
                 i, 
                 confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
            