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
    counts = np.zeros(shape=(unique_outcomes.size, ))
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

    return np.sum(-p*np.log2(p))


def gini_impurity(outcomes, unique_outcomes):
    """
    :param outcomes: numpy 1d-array
    :param unique_outcomes: numpy 1d-array
    :return: computed gini impurity
    """
    if not outcomes.size > 0 or not unique_outcomes.size > 0:
        raise IndexError("Inputs cannot be empty arrays")
    return 1-sum([(np.sum(outcomes == unique_outcomes[idx])/outcomes.size)**2
                 for idx in range(unique_outcomes.size)])


class Branch:
    """
    """

    def __init__(self):
        self.probabilities = {}
        self.branches = []


def grow_tree(data, outcomes, tree_type="cart"):
    """
    :param data:
    :param outcomes:
    :param tree_type:
    :return:
    """
    if data.shape[0] != outcomes.size:
        raise Exception("Mismatch in the number of data samples and outcomes")

    # TODO: need to add logic to create metadata for data and pass along the
    # list of variables
    return _grow_tree(data, outcomes, TYPES[tree_type])


def _grow_tree(data, outcomes, impurity_function):
    """
    :param data:
    :param outcomes:
    :param impurity_function:
    :return:
    """
    pass
