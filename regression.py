"""
file: regression.py
author: Kantha Girish G
description: Implementation of LinearRegression for line fitting and Logistic
Regression for classification
"""

import numpy as np

from util import get_squared_error, split_data, sigmoid \
    , cross_entropy_loss, error_rate


class LinearRegression:
    """
    Implementation of Regression, line fitting. The implementation fits a linear model to
    the given data points. Data can be single or multi-featured. Gradient descent with least
    squares error is used for parameter learning. The implementation adds intercept parameter.
    """

    def __init__(self, W=None):
        self.Weights = W

    def train(self, X, Y, alpha=0.1e-4, epochs=500, W_init=None):
        """
        :param X: numpy 1D/2D array of data, parameter(s) on which Y is
        dependent on
        :param Y: numpy 1D array, dependent values
        :param alpha: learning rate
        :param epochs: number of steps to repeat for convergence
        :return: a python list of errors computed for all the epochs.

        This method fits a linear model for the given set of data X to best represent the
        target Y.
        """

        if X.ndim == 1:
            X.shape = [X.size, 1]

        m = X.shape[0]
        print(X.shape)
        XX = np.hstack((np.ones(shape=(m, 1)), X))

        Y.shape = [Y.size, 1]

        # initialize random values for weights
        if W_init is None:
            W = np.zeros(shape=(XX.shape[1], 1))
        else:
            W = W_init
            W.shape = (W.size, 1)

        # Perform gradient descent
        best_error = np.inf
        errors = []
        print(XX.shape)
        print(Y.shape)
        print(W.shape)
        for step in range(epochs):
            grad = 2 / Y.size * XX.transpose().dot((XX.dot(W) - Y))
            W -= alpha * grad
            error = get_squared_error(Y, XX, W)
            errors.append(error)
            if error < best_error:
                best_error = error
                self.Weights = np.copy(W)

        return errors

    def predict(self, X):
        """
        :param X: numpy 1D/2D array of data for which prediction of
                regression line is required.
        :return: numpy 1D array of values predicted for each row in X.

        This method uses trained parameters to predict/estimate target values corresponding
        to the values in data X.
        """

        if X.ndim == 1:
            X.shape = [X.size, 1]

        m = X.size
        XX = np.hstack((np.ones(shape=(m, 1)), X))

        return XX.dot(self.Weights)


class LogisticRegression:
    """
    Implementation of Logistic Regression classifier using gradient descent optimization for
    parameter learning. The implementation uses a linear model and adds intercept
    automatically.
    """

    def __init__(self, W=None, b=None):
        self.W = W
        self.bias = b

    def train(self, X, Y, step_size=10e-5, epochs=10000, validation_frac=0.1):
        """
        :param X: data, numpy 2D array
        :param Y: labels, numpy 1D array of target labels, only two possible classes.
        :param step_size: size of the step for gradient descent
        :param epochs: number of max iterations
        :param validation_frac: Fraction of data to use for validation,
                default: 10%
        :return: best validation error

        This method fits a model with a linear decision boundary to classify the data
        samples X by learning parameters using Gradient Descent optimizer. The training data X
        is further split into training and validation data. The parameters corresponding to
        lowest error on validation data over the epochs, are stored as trained parameters.
        """
        # Validation data set extracted from the training data
        Xvalid, Yvalid, X, Y = split_data(X, Y, validation_frac)
        N, D = X.shape

        print("Training data size: ({}, {})".format(N, D))

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

        for i in range(epochs):

            if i % 100 == 0:
                print(i)

            # Do forward propagation to calculate P(Y|X)
            pY = sigmoid(X.dot(W) + bias)

            # Perform gradient descent
            W -= step_size * (X.transpose().dot(pY - Y) / N).reshape(D, 1)
            bias -= step_size * np.mean(pY - Y)

            # Compute the sigmoid costs and append to array costs
            # Check to set best_validation_error
            cost = cross_entropy_loss(Y, pY)
            costs.append(cost)

            # Using the validation data, compute P(Y|X_valid)
            pYvalid = sigmoid(Xvalid.dot(W) + bias)
            error = error_rate(Yvalid, np.round(pYvalid))
            errors.append(error)

            if error < best_validation_error:
                best_validation_error = error
                self.W = np.copy(W)
                self.bias = bias

        print("\n")

        return costs, errors

    def predict(self, X, return_probs=False):
        """
        :param X: numpy 2D array (M x D) of data for which predictions
                are to be made.
        :param return_probs: Boolean indicating whether to return predicted probabilities
        :return: numpy 1D array of class labels[, predicted probabilities ]

        This method returns the predicted class labels for the given data samples X using
        trained parameters.
        """
        pY = sigmoid(X.dot(self.W) + self.bias)
        if return_probs:
            return np.round(pY), pY
        else:
            return np.round(pY)

    def score(self, X, Y):
        """
        :param X: numpy 2D array (M x D) of data for which predictions
                are to be made.
        :param Y: numpy 1D array (M x 1) of true labels corresponding to data in
                X
        :return: Accuracy of prediction on X using the trained model
        """
        cY = self.predict(X)
        return 1 - error_rate(Y, cY)
