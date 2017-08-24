import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer

from trees import ClassificationTree
from util import accuracy, kde, split_data


def get_adult_data():
    """
    :return: data, labels

    Function to load cleaned Adult data set.
    """
    df = pd.read_csv('Adult.csv', delimiter=',', header=None, index_col=None)
    columns = df.columns
    X = df.as_matrix(columns[:-1])
    Y = df.as_matrix([columns[-1]])
    return X, Y


def test_tree_classifier():
    """
    :return: None

    Function to test decision tree classifier
    """
    # X, Y = get_adult_data()
    # attr_types = [int for _ in range(X.shape[1])]

    data = load_breast_cancer()
    X = data.data
    Y = data.target.reshape(data.target.size)

    attr_types = [float for _ in range(X.shape[1])]

    Xtrain, Ytrain, Xtest, Ytest = split_data(X, Y, 0.8)
    model = ClassificationTree()
    print("Training..")
    model.train(Xtrain, Ytrain, attr_types)
    model.prune_tree(Xtrain, Ytrain)
    cY = model.predict(Xtest)
    print("Accuracy: {}".format(accuracy(Ytest, cY)))

    clf = tree.DecisionTreeClassifier()
    clf.fit(Xtrain, Ytrain.reshape(Ytrain.size))
    cY = clf.predict(Xtest)
    print("Scikit accuracy: ".format(accuracy(Ytest, cY)))


def test_kde():
    """
    :return: None

    Function to test Kernel Density Estimation
    """
    data1 = norm.rvs(10, 2, size=100)
    data2 = norm.rvs(0, 3, size=100)
    data3 = norm.rvs(20, 2, size=100)
    data4 = norm.rvs(30, 3, size=100)
    data5 = norm.rvs(40, 3, size=100)
    data = np.concatenate((data1, data2, data3, data4, data5))

    inflexion_points, range_, kdest = kde(data, return_kde=True)

    print(inflexion_points)
    plt.subplot(211)
    plt.hist(data, bins=100, color='b')

    plt.subplot(212)
    plt.plot(range_, kdest, color='g', linewidth=2)

    plt.plot(inflexion_points[:, 0], inflexion_points[:, 1], color='k',
             linewidth=2)
    plt.show()


if __name__ == '__main__':
    test_tree_classifier()