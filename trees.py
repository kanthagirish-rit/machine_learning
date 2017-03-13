__author__ = "Kantha Girish"

import numpy as np
import matplotlib.pyplot as plt

from util import count

TYPES = {}


def entropy_impurity(outcomes, unique_outcomes):
    """
    :param outcomes: numpy 1d-array
    :param unique_outcomes: numpy 1d-array
    :return: computed entropy impurity of the outcomes
    """
    if not outcomes.size > 0 or not unique_outcomes.size > 0:
        raise IndexError("Inputs cannot be empty arrays")
    counts = np.zeros(shape=(unique_outcomes.size,))
    for idx in range(unique_outcomes):
        counts[idx] = count(outcomes, unique_outcomes[idx])
    return entropy(counts)


def entropy(counts):
    """
    :param counts: numpy 1d-array
    :return: entropy
    """
    if (np.where(counts < 0)[0]).size > 0:
        raise ValueError("counts cannot be negative")

    p = counts / np.sum(counts)

    # replace zero counts with 1 to avoid log(0)
    p[np.where(p == 0)[0]] = 1

    return np.sum(-p * np.log2(p))


def gini_impurity(outcomes, unique_outcomes):
    """
    :param outcomes: numpy 1d-array
    :param unique_outcomes: numpy 1d-array
    :return: computed gini impurity
    """
    if not outcomes.size > 0 or not unique_outcomes.size > 0:
        raise IndexError("Inputs cannot be empty arrays")
    return 1 - sum([(np.sum(outcomes == unique_outcomes[idx]) / outcomes.size) ** 2
                    for idx in range(unique_outcomes.size)])


class Tree:
    """
    """
    __slots__ = ["root", "impurity_function", "data", "outcomes",
                 "outcome_values", "rows", "columns"]

    def __init__(self, impurity_function, data, outcomes, outcome_values):
        self.root = None
        self.impurity_function = impurity_function
        self.data = data
        self.outcomes = outcomes
        self.outcome_values = outcome_values
        self.rows, self.columns = self.data.shape

    def _grow(self, rows, columns, value=None):
        """
        :param rows:
        :param columns:
        :param value:
        :return:
        """

        # Create node and populate fields
        node = Node(value)
        previous_prob = 0
        for outcome in self.outcome_values:
            node.probabilities[outcome] = np.where(self.outcomes[rows] == outcome)[0]/rows.size
            if node.probabilities[outcome] > previous_prob:
                node.p_class = outcome

        if self.root is None:
            self.root = node

        # Calculate the impurity of the current node
        node_impurity = self.impurity_function(self.outcomes[rows, :], self.outcome_values)

        children_impurity = 0
        information_gains = np.zeros(shape=(columns.size, 1))
        for index in range(columns):
            unique_values = np.unique(self.data[rows, columns[index]])
            for value in unique_values:
                value_indices = np.where(self.data[rows, columns[index]] ==
                                         value)[0]
                value_probability = value_indices.size / rows.size
                value_impurity = self.impurity_function(self.outcomes[rows][value_indices],
                                                        self.outcome_values)
                children_impurity += value_probability * value_impurity
            information_gains[index] = node_impurity - children_impurity

        max_gain = np.amax(information_gains)
        gain_column_index = np.argmax(information_gains)

        node.next_node = columns[gain_column_index]

    def grow(self):
        """
        :return:
        """
        pass


class Node:
    """
    """
    __slots__ = ["value", "branches", "probabilities", "next_node", "is_leaf", "p_class"]

    def __init__(self, value):
        self.value = value
        self.branches = []
        self.probabilities = {}
        self.next_node = None
        self.is_leaf = False
        self.p_class = None


def grow_tree(data, outcomes, tree_type="cart", parameter_values=None,
              outcome_values=None):
    """
    :param data:
    :param outcomes:
    :param tree_type:
    :param parameter_values:
    :param outcome_values:
    :return:
    """
    if data.shape[0] != outcomes.size:
        raise Exception("Mismatch in the number of data samples and outcomes")

    if parameter_values is None:
        parameter_values = []
        for param in range(data.shape[1]):
            parameter_values.append(np.unique(data[:, param]))

    if outcome_values is None:
        outcome_values = np.unique(outcomes)
    if data.shape[1] != len(parameter_values):
        raise Exception("Number of possible parameters is not equal to number "
                        "of parameters in data")

    # list of variables
    tree = Tree(TYPES[tree_type], data, outcomes, outcome_values)
    tree.grow()
