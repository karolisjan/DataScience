#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:55:10 2018

@author: karolis
"""

import numpy as np


def gini_impurity(groups, labels):
    '''
        Returns a Gini Impurity value weighted by the size of each class.
    '''
    total_size = sum([len(group) for group in groups])
    cost = 0
    
    for group in groups:
        
        group_size = len(group)
        group_score = 0
        
        if group_size == 0:
            continue
        
        for label in labels:
            probability = sum(group == label) / group_size
            group_score += probability * probability
            
        cost += (1 - group_score) * (group_size / total_size)
    
    return cost


class RandomForest():
    
    def __init__(
        self, 
        cost_function=gini_impurity,
        num_trees=10,
        num_features='auto',
        max_depth=10,
        sample_ratio=1,
        min_samples_split=1,
    ):            
        assert num_features == 'auto' or type(num_features) is int
        self.num_features = num_features
        
        assert sample_ratio > 0.0 and sample_ratio <= 1.0, \
        'sample_ratio must greater than 0 and less than or equal to 1.'
        self.sample_ratio = sample_ratio
        
        self.num_trees = num_trees
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split
        
        self.cost_function = cost_function
            
    def __get_terminal(self, y):
        '''
            Returns a label with the highest count.
        '''
        return max(set(y), key=y.tolist().count)
    
    def __split(self, feature, value, X, y):
        '''
            Splits the X and y into two based on a feature and its value.
        '''
        split_mask = X[:, feature] < value
                
        # Left and Right splits
        return X[split_mask], y[split_mask], X[~split_mask], y[~split_mask]
        
    def __get_node(self, X, y):
        '''
            Returns a best a split.
        '''
        features = np.random.choice(X.shape[1], self.num_features)
        
        best_feature = None
        best_val = None
        best_cost = np.inf
        best_groups = [[None, None], [None, None]]
        labels = set(y)
        
        for feature in features:
            for x in X:
                
                X_left, y_left, X_right, y_right = self.__split(feature, x[feature], X, y)
                cost = self.cost_function([y_left, y_right], labels)
                
                if cost < best_cost:
                    best_feature = feature
                    best_val = x[feature]
                    best_cost = cost
                    best_groups = [[X_left, y_left], [X_right, y_right]]
        
        return {
            'feature': best_feature,
            'val': best_val,
            'X_left': best_groups[0][0],
            'y_left': best_groups[0][1],
            'X_right': best_groups[1][0],
            'y_right': best_groups[1][1]
        } 
            
    def __add_node(self, node, depth):
        '''
            Adds nodes to the tree recursively until the value of depth reaches 0.
        '''
        
        X_left, y_left = node['X_left'], node['y_left']
        X_right, y_right = node['X_right'], node['y_right']
         
        if y_left is None or y_left.size == 0:
            node['left'] = node['right'] = self.__get_terminal(y_right)
            del node['X_left'], node['y_left'], node['X_right'], node['y_right']
            return
        
        if y_right is None or y_right.size == 0:
            node['left'] = node['right'] = self.__get_terminal(y_left)
            del node['X_left'], node['y_left'], node['X_right'], node['y_right']
            return
        
        if depth <= 0:
            node['left'] = self.__get_terminal(y_left)
            node['right'] = self.__get_terminal(y_right)
            del node['X_left'], node['y_left'], node['X_right'], node['y_right']
            return
        
        if len(y_left) <= self.min_samples_split:
            node['left'] = self.__get_terminal(y_left)
        else:
            node['left'] = self.__get_node(X_left, y_left)
            self.__add_node(node['left'], depth - 1)
            
        if len(y_right) <= self.min_samples_split:
            node['right'] = self.__get_terminal(y_right)
        else:
            node['right'] = self.__get_node(X_right, y_right)
            self.__add_node(node['right'], depth - 1)
            
        del node['X_left'], node['y_left'], node['X_right'], node['y_right']
    
    def __make_tree(self, X, y):
        '''
            Creates a decision tree.
        '''
        root = self.__get_node(X, y)
        self.__add_node(root, self.max_depth)
        return root
    
    def __get_sample(self, X, y):
        sample_mask = np.random.rand(X.shape[0]) <= self.sample_ratio
        return X[sample_mask], y[sample_mask]
    
    def fit(self, X, y):        
        assert len(X) == len(y), \
        'The length of X ({}) does not match the length of y ({}).'.format(len(X), len(y))
        
        X = np.array(X).reshape(len(X), -1)
        
        if type(self.num_features) is int:
            assert self.num_features <= X.shape[1], \
            'The number of features available ({}) is less than the number of features specified ({}).'.format(X.shape[1], self.num_features)
        else:
            self.num_features = int(np.sqrt(X.shape[1]))
            
        try:
            labels = set(y)
        except BaseException:
            raise ValueError('y needs to be a column vector.')
        
        self.forest = []
        
        for i in range(self.num_trees):
            X_sample, y_sample = self.__get_sample(X, y)
            tree = self.__make_tree(X_sample, y_sample)
            self.forest.append(tree)
            
        return self
        
    def __predict_tree(self, node, x):
        if x[node['feature']] < node['val']:
            return node['left'] if not isinstance(node['left'], dict) else self.__predict_tree(node['left'], x)
        return node['right'] if not isinstance(node['right'], dict) else self.__predict_tree(node['right'], x)

    def __predict_forest(self, x):
        predictions = [self.__predict_tree(tree, x) for tree in self.forest]
        return max(set(predictions), key=predictions.count)

    def predict(self, X):
        return np.array([self.__predict_forest(x) for x in X])