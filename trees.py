"""
file: dtree.py
author: Kantha Girish
description: Implements decision tree with entropy information gain
"""
import numpy as np

from util import entropy, probabilities


class DTree:
    """
    Implementation of Decision tree
    """
    def __init__(self):
        self.root = None

    def _gain(self, X, Y):
        """
        :param X: numpy 1D array, data samples
        :param Y: numpy 1D array, class labels
        :return: computed information gain
        """
        entropy_total = entropy(Y)
        entropy_subsets = 0

        unique_values = np.unique(X)
        for value in unique_values:
            subset = Y[np.where(X == value)[0]]
            entropy_subsets += (subset.size / Y.size) * entropy(subset)

        return entropy_total - entropy_subsets

    def _best_attribute(self, X, Y, attributes):
        """
        :param X: numpy 2D array, data samples
        :param Y: numpy 1D array, class labels
        :param attributes: list of attributes in the data
        :return: The attribute with highest information gain
        """
        best_gain = -np.inf
        best_attr = None
        for attr in attributes:
            g = self._gain(X[:, attr], Y)
            if g > best_gain:
                best_gain = g
                best_attr = attr
        return best_attr

    def fit(self, X, Y, attributes, depth, value=None):
        """
        :param X: numpy 2D array, data samples
        :param Y: numpy 1D array, class labels
        :param attributes: list of attributes in the data
        :param depth: maximum depth of the tree
        :param value: possible value of the attribute for the current node/branch
        :return: a decision tree built to the chosen depth
        """
        self.root = self._grow_tree(X, Y, attributes, depth, value=None)

    def _grow_tree(self, X, Y, attributes, depth, value=None):
        # Construct a node
        node = Node()
        node.probs = probabilities(Y)
        node.branch_value = value

        # Stop criteria
        if depth == 0 or entropy(Y) == 0 or len(attributes) == 0:
            return node

        # Find the best attribute
        attr = self._best_attribute(X, Y, attributes)
        attributes.remove(attr)
        node.next_attr = attr

        # Recurse to construct children for the node
        values = np.unique(X[:, attr])
        for val in values:
            subset_indices = np.where(X[:, attr] == val)[0]
            node.branches[val] = self._grow_tree(X[subset_indices, :], Y[subset_indices]
                                                 , attributes, depth - 1, val)
        return node


class Node:
    """
    Node class to build a decision tree
    """
    def __init__(self):
        self.probs = {}
        self.branches = {}
        self.next_attr = None
        self.branch_value = None

    def __str__(self):
        return "answer: " + str(self.branch_value) + "\n" + "p: " + str(self.probs) + "\n" + \
               "next question: " + str(self.next_attr)

