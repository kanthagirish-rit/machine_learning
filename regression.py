"""
file: regression.py
author: Kantha Girish G
description: Implementation of Regression for line/Curve fitting and Logistic
Regression for classification
"""

import numpy as np

from util import get_squared_error, shuffle, split_data, sigmoid \
    , cross_entropy_loss, error_rate


class Regression:
    """
    Implementation of Regression, line/curve fitting
    """

    def __init__(self, W=None, polynomial_order=None):
        self.Weights = W
        self.polynomial_order = polynomial_order

    def train(self, X, Y, polynomial_order, alpha=0.1e-4, epochs=500,
            W_init=None):
        """
        :param X: numpy 1D array or list, parameter on which Y is dependent on
        :param Y: numpy 1D array or list, dependent values
        :param polynomial_order: order of the polynomial to use for
                function fitting
        :param alpha: learning rate
        :param epochs: number of steps to repeat for convergence
        :return: a python list of errors computed for all the epochs.
        """
        self.polynomial_order = polynomial_order

        if type(X) == list:
            X = np.array(X)
        if type(Y) == list:
            Y = np.array(Y)

        m = X.size
        n = polynomial_order + 1
        XX = np.ones(shape=(m, n))

        Y.shape = (Y.size, 1)

        for i in range(1, n):
            XX[:, i] = X ** i

        # initialize random values for weights
        if W_init is None:
            W = np.zeros(shape=(n, 1))
        else:
            W = W_init
            W.shape = (W.size, 1)

        # Perform gradient descent
        best_error = np.inf
        errors = []
        for step in range(epochs):
            grad = self.get_gradients(Y, XX, W)
            W -= alpha * grad
            error = get_squared_error(Y, XX, W)
            errors.append(error)
            if error < best_error:
                best_error = error
                self.Weights = np.copy(W)

        return errors

    def get_gradients(self, Y, XX, W):
        """
        :param Y: numpy 1D array dependent values
        :param XX: numpy 2D array of data
        :param W: numpy 1D array of co-efficients
        :return: averaged gradients for the current model
        """
        return 2 / Y.size * XX.transpose().dot((XX.dot(W) - Y))

    def predict(self, X):
        """
        :param X:
        :param model:
        :return:
        """
        if type(X) == list:
            X = np.array(X)

        m = X.size
        n = self.Weights.size
        XX = np.ones(shape=(m, n))

        for i in range(1, n):
            XX[:, i] = X ** i

        return XX.dot(self.Weights)


class LogisticRegression:
    """
    Implementation of Logistic Regression classifier
    """

    def __init__(self, W=None, b=None):
        self.W = W
        self.bias = b

    def train(self, X, Y, step_size=10e-5, epochs=10000, validation_frac=0.1):
        """
        :param X: data, numpy 2D array
        :param Y: labels, numpy 1D array
        :param step_size: size of the step for gradient descent
        :param epochs: number of max iterations
        :param validation_frac: Fraction of data to use for validation,
                default: 10%
        :return: best validation error
        """
        # Validation data set extracted from the training data
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid, X, Y = split_data(X, Y, validation_frac)
        N, D = X.shape

        # Make sure Y and Yvalid are column vectors
        Y.shape = [Y.size, 1]
        Yvalid.shape = [Yvalid.size, 1]

        # Initialize the weights W and the bias b to zero
        W = np.zeros(shape=(D, 1), dtype=np.float32)
        bias = np.zeros(shape=1, dtype=np.float32)

        # Perform Gradient Descent over defined epochs to learn weights
        costs = []
        errors = []
        best_validation_error = 1
        min_cost = np.inf

        for i in range(epochs):

            if i % 100 == 0:
                print(".", end="")

            # Do forward propagation to calculate P(Y|X)
            pY = sigmoid(X.dot(W) + bias)

            # Perform gradient descent
            W -= step_size * (X.transpose().dot(pY - Y) / N).reshape(D, 1)
            bias -= step_size * np.mean(pY - Y)

            # Using the validation data, compute P(Y|X_valid)
            # Compute the sigmoid costs and append to array costs
            # Check to set best_validation_error
            pYvalid = sigmoid(Xvalid.dot(W) + bias)
            cost = cross_entropy_loss(Yvalid, pYvalid)
            costs.append(cost)

            pYvalid = np.round(pYvalid)
            error = error_rate(Yvalid, pYvalid)
            errors.append(error)

            if error < best_validation_error:
                best_validation_error = error
                self.W = np.copy(W)
                self.bias = bias

        print("\n")

        return errors

    def predict(self, X):
        """
        :param X: numpy 2D array (M x D) of data for which predictions are to be made.
        :return: predicted probabilities pY and class labels
        """
        pY = sigmoid(X.dot(self.W) + self.bias)
        return pY, np.round(pY)

