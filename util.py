
import numpy as np
from scipy.stats import norm


MIN_FLOAT = np.finfo(np.float64).eps


def kde(data, resolution=None, return_kde=False):
    """
    :param data:
    :param resolution:
    :param return_kde:
    :return:
    """
    data.shape = (data.size, 1)

    if resolution is None:
        resolution = (np.unique(data)).size/1000

    # Calculate standard deviations using the Silverman's rule of thumb,
    # but restricting the denominator to cube root to have less smoothened
    # curves
    kernel_sigma = 1.06 * np.sqrt(np.var(data)) / (data.size**(1/3))

    range_ = np.arange(start=np.min(data)-2, stop=np.max(data)+2,
                       step=resolution)
    range_.shape = (range_.size, )

    # Initialize an empty array to compute Kernel Density Estimation and
    # compute KDE
    kdest = np.zeros(shape=range_.shape)
    for idx in range(data.size):
        kdest += norm.pdf(range_, data[idx, 0], kernel_sigma)

    # Compute slope at each point by subtracting it with the previous point
    slope = np.zeros(shape=(kdest.size-1, ))
    for idx in range(slope.size):
        slope[idx] = kdest[idx+1] - kdest[idx]

    # Compute sign of slope points in order to find peaks and troughs
    signs = np.sign(slope)

    # Multiply consecutive signs to arrive at points which are peaks and troughs
    inflexions = np.zeros(shape=(signs.size-1, ))
    for idx in range(inflexions.size):
        inflexions[idx] = signs[idx+1] * signs[idx]

    inf_idx = np.where(inflexions == -1)[0]

    # Store the inflexion points in a 2D array and return it
    inflexion_points = np.zeros(shape=(inf_idx.size+2, 2))

    inflexion_points[1:-1, 0] = range_[inf_idx, ]
    inflexion_points[1:-1, 1] = kdest[inf_idx, ]

    inflexion_points[0, 0] = range_[0]
    inflexion_points[-1, 0] = range_[-1]

    inflexion_points[0, 1] = kdest[0]
    inflexion_points[-1, 1] = kdest[-1]

    if return_kde:
        return inflexion_points, range_, kdest
    else:
        return inflexion_points


def categorize_parameter(data, num_values, inflexion_points=None):
    """
    :param data:
    :param num_values:
    :param inflexion_points:
    :return:
    """

    # Find out Inflexion points if it is not provided
    if inflexion_points is None:
        inflexion_points = kde(data, num_values/1000)
    categories = inflexion_points[::2, 0]
    categories.reshape(1, categories.size)


def probabilities(Y):
    """
    :param Y: numpy 1D array, class labels
    :return: computed probabilities for each class in Y as a dictionary
            p = {
                    class1: probability_value,
                    class2: probability_value
                    .
                    .
                }
    """
    unique_labels = np.unique(Y)
    p = {}
    for val in unique_labels:
        p[val] = (np.where(Y == val)[0]).size / Y.size
    return p


def entropy(Y):
    """
    :param Y: numpy 1D array, class labels
    :return: entropy/information
    """

    p = probabilities(Y)
    return -np.sum([i * np.log2(i) for i in p.values()])


def get_squared_error(Y, XX, W):
    """
    :param Y: numpy 1D array dependent values
    :param XX: numpy 2D array of data
    :param W: numpy 1D array of co-efficients
    :return: averaged least squares error
    """
    return np.mean((Y - XX.dot(W))**2)


def shuffle(X, Y):
    """
    :param X:
    :param Y:
    :return:
    """
    if X.shape[0] != Y.size:
        raise Exception("Number of rows in X and Y should be equal")
    index = np.arange(Y.size)
    np.random.shuffle(index)
    return X[index, :], Y[index]


def split_data(X, Y, split_frac):
    """
    :param X: numpy 1D/2D array of data
    :param Y: numpy 1D array of class labels corresponding to the data in X.
            Number of of rows in Y should be same as in X
    :param split_frac: split fraction between (0,1)
    :return: X_left_half, Y_left_half, X_right_half, Y_right_half
    """
    if X.shape[0] != Y.size:
        raise Exception("Number of rows in X and Y should be equal")
    X, Y = shuffle(X, Y)
    split_index = int(np.round(Y.size * split_frac))
    print(split_index)
    return X[:split_index, :], Y[:split_index], X[split_index:, :], \
        Y[split_index:]


def sigmoid(X):
    """
    :param X: numpy 1D/2D array of data
    :return: sigmoid of X
    """
    return (1 / (1 + np.exp(-X))).reshape(X.size, 1)


def cross_entropy_loss(T, Y):
    """
    :param T: numpy 1D array of target labels
    :param Y: numpy 1D array of predicted probabilities
    :return:
    """
    Y[Y == 1] = 1 - MIN_FLOAT
    Y[Y == 0] = MIN_FLOAT

    # Make sure shapes match
    T.shape = [T.size, 1]
    Y.shape = [Y.size, 1]
    return -np.mean(T*np.log(Y) + (1-T)*np.log(1-Y))


def error_rate(targets, predictions):
    """
    :param targets: numpy 1D array of target labels
    :param predictions: numpy 1D array of predicted labels
    :return: error rate in prediction
    """
    targets.shape = [targets.size, 1]
    predictions.shape = [predictions.size, 1]
    return np.mean(targets != predictions)


def accuracy(targets, predictions):
    """
    :param targets: numpy 1D array of target labels
    :param predictions: numpy 1D array of predicted labels
    :return: error rate in prediction
    """
    targets.shape = [targets.size, 1]
    predictions.shape = [predictions.size, 1]
    return np.mean(targets == predictions)


def get_best_class_prob(probs):
    """
    :param probs: a dictionary containing class labels and probabilities as key value pairs
            as shown below.
            probs = {
                "class_1" : prob1,
                "class_2" : prob2
            }
    :return:
    """
    prob = -np.inf
    class_ = None
    for key in probs:
        if prob < probs[key]:
            prob = probs[key]
            class_ = key
    return class_, prob
