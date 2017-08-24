"""
file: dtree.py
author: Kantha Girish
description: Implements decision trees and forests for classification and regression.
"""
import numpy as np

from util import entropy, probabilities, get_best_class_prob


L_BRANCH = "left-br"
R_BRANCH = "right-br"

class ClassificationTree:
    """
    Implementation of Decision tree using entropy for computing information gain. The
    implementation handles both discrete and continuous attributes. This is done by passing
    in a python list of data-types for attributes. Type `int` indicates the attribute is
    discrete/categorical and type `float` indicates that the attribute is continuous. The
    trained model is persisted as a class member. This implementation is closest to `C4.5`.
    """
    def __init__(self):
        self.root = None
        self.attr_dtypes = None

    def _gain(self, X, Y, attr):
        """
        :param X: numpy 1D array, data samples
        :param Y: numpy 1D array, class labels
        :return: computed information gain

        This method computes information gain
        """
        entropy_total = entropy(Y)

        unique_values = np.unique(X)
        if len(unique_values) == 1:
            return 0, None
        if self.attr_dtypes[attr] == int:
            entropy_subsets = 0
            for value in unique_values:
                subset = Y[np.where(X == value)[0]]
                entropy_subsets += (subset.size / Y.size) * entropy(subset)
            return entropy_total - entropy_subsets
        elif self.attr_dtypes[attr] == float:
            best_gain = 0
            split_value = None
            for value in unique_values[1:]:
                left_subset = Y[np.where(X < value)[0]]
                right_subset = Y[np.where(X >= value)[0]]
                entropy_subsets = (left_subset.size / Y.size) * entropy(left_subset) + \
                                  (right_subset.size / Y.size) * entropy(right_subset)
                if entropy_total - entropy_subsets > best_gain:
                    best_gain = entropy_total - entropy_subsets
                    split_value = value
            return best_gain, split_value

    def _best_attribute(self, X, Y, attributes):
        """
        :param X: numpy 2D array, data samples
        :param Y: numpy 1D array, class labels
        :param attributes: list of attributes in the data
        :return: The attribute with highest information gain
        """
        best_gain = -np.inf
        best_attr = None
        best_split = None
        for attr in attributes:
            split_val = None
            if self.attr_dtypes[attr] == int:
                g = self._gain(X[:, attr], Y, attr)
            elif self.attr_dtypes[attr] == float:
                g, split_val = self._gain(X[:, attr], Y, attr)
            if g > best_gain:
                best_gain = g
                best_attr = attr
                if self.attr_dtypes[attr] == float:
                    best_split = split_val
        return best_attr, best_gain, best_split

    def train(self, X, Y, attr_dtypes, depth=None):
        """
        :param X: numpy 2D array, data samples
        :param Y: numpy 1D array, class labels
        :param depth: maximum depth of the tree
        :param attr_dtypes: a python list containing data-types of attributes/columns in X
                            as described below.

                            int - column has categorical values
                            float - continuous values
        :return: a decision tree built to the chosen depth

        This method calls the recursive method which grows a decision tree using the given
        data. Entropy is used as a measure for information gain. The model handles both
        continuous and discrete/categorical attributes.
        """
        self.attr_dtypes = attr_dtypes
        self.root = self._grow_tree(X, Y, list(range(X.shape[1])), depth, value=None)

    def _grow_tree(self, X, Y, attributes, depth, value=None):
        """
        :param X: numpy 2D array, data samples
        :param Y: numpy 1D array, class labels
        :param attributes: list of attributes in the data
        :param depth: maximum depth of the tree
        :param value: possible value of the attribute for the current node/branch
        :return: a constructed node which has branches pointing further nodes

        This method grows a decision tree by recursively calling itself.
        """
        # Construct a node
        node = Node()
        node.probs = probabilities(Y)
        node.class_, _ = get_best_class_prob(node.probs)
        node.branch_value = value

        # Stop criteria
        if depth == 0 or entropy(Y) == 0 or len(attributes) == 0:
            return node

        # Find the best attribute
        attr, node.gain, split_value = self._best_attribute(X, Y, attributes)
        node.next_attr = attr

        if node.gain == 0:
            return node

        if depth:
            depth -= 1

        # Recurse to construct child nodes. The branches are built based on the type of the
        # attribute.
        # If the attribute is continuous, only two branches are formed (left-br,right-br)
        # If the attribute is discrete, a branch is created for each possible value of the
        # attribute.
        if self.attr_dtypes[attr] == int:
            attributes.remove(attr)
            values = np.unique(X[:, attr])
            for val in values:
                subset_indices = np.where(X[:, attr] == val)[0]
                node.branches[val] = self._grow_tree(X[subset_indices, :], Y[subset_indices]
                                                     , attributes, depth, val)
        elif self.attr_dtypes[attr] == float:
            node.split_value = split_value
            left_subset = np.where(X[:, attr] < split_value)[0]
            node.branches[L_BRANCH] = self._grow_tree(X[left_subset, :], Y[left_subset]
                                                       , attributes, depth, None)
            right_subset = np.where(X[:, attr] >= split_value)[0]
            node.branches[R_BRANCH] = self._grow_tree(X[right_subset, :], Y[right_subset]
                                                        , attributes, depth, None)
        return node

    def predict(self, X, return_probs=False):
        """
        :param X: numpy 2D array of data
        :param return_probs: Boolean indicating whether to return predicted class
                probabilities or not.
        :return: numpy array of predicted class labels [, numpy array of predicted class
                probabilities]

        This method returns the predicted classes[, probabilities] for the given data
        samples using the trained decision tree.
        """
        cY = np.zeros(shape=(X.shape[0], 1))
        probs = np.zeros(shape=(X.shape[0], 1))

        # Get the class labels for each sample by traversing down the tree
        for row in range(X.shape[0]):
            current = self.root
            while current.next_attr is not None:
                value = X[row, current.next_attr]
                if current.split_value is None:
                    if value not in current.branches:   # branch is pruned
                        break
                    current = current.branches[value]
                else:
                    if value < current.split_value and L_BRANCH in current.branches:
                        current = current.branches[L_BRANCH]
                    elif value >= current.split_value and R_BRANCH in current.branches:
                        current = current.branches[R_BRANCH]
                    else:       # branch is pruned
                        break
            cY[row, 0], probs[row, 0] = get_best_class_prob(current.probs)
        if return_probs:
            return cY, probs
        else:
            return cY

    def prune_tree(self, X, Y):
        """
        :param X: numpy 2D array of data samples
        :param Y: numpy 1D array of class labels
        :return: None

        This method implements Reduced error pruning technique on the decision that is
        built. The tree is pruned using the given data samples.
        """
        for i in range(Y.size):
            self._create_re_tree(X[i, :], Y[i])
        self._prune_tree(self.root)

    def _prune_tree(self, node):
        """
        :param node: current node of the decision tree
        :return: classification error of the current node

        This method prunes the tree by recursively calling itself. This is a helper method
        used by prune_tree() to prune the decision tree that is built.
        """
        removable_branches = []
        for key, value in node.branches.items():
            subtree_error = self._prune_tree(value)
            if node.error <= subtree_error:
                removable_branches.append(key)

        for branch in removable_branches:
            del(node.branches[branch])
        return node.error

    def _create_re_tree(self, X, Y):
        """
        :param X: numpy 1D array, one instance of data
        :param Y: scalar, class corresponding to the instance X
        :return: None

        This method modifies the tree created to include classification errors at each node.
        The error would be used to Prune the tree.
        """
        node = self.root
        while node.next_attr is not None:
            if node.class_ != Y:
                node.error += 1
            branch_value = X[node.next_attr]
            if node.split_value is None:
                node = node.branches[branch_value]
            else:
                if branch_value < node.split_value:
                    node = node.branches[L_BRANCH]
                else:
                    node = node.branches[R_BRANCH]


# TODO: Implement random forests
# TODO: Implement a regression tree

class Node:
    """
    Node class to build a decision tree
    """
    def __init__(self):
        self.probs = {}
        self.branches = {}
        self.next_attr = None
        self.branch_value = None
        self.gain = 0
        self.split_value = None
        self.error = 0
        self.class_ = None
