__author__ = "Kantha Girish"

import numpy as np
from scipy.stats import norm


def kde(data, resolution):
    """
    :param data:
    :param resolution:
    :return:
    """
    data.shape = (data.size, 1)

    # Calculate standard deviations using the Silverman's rule of thumb
    kernel_sigma = 1.06 * np.sqrt(np.var(data)) / (data.size**5)

    range_ = np.arange(start=np.min(data)-2, stop=np.max(data)+2,
                       step=resolution)
    range_.shape = (range_.size, )

    # Initialize an empty array to compute Kernel Density Estimation and
    # compute KDE
    kdest = np.zeros(shape=range_.shape)
    for point in data:
        kdest += norm.pdf(range_, point, kernel_sigma)

    # Compute slope at each point by subtracting it with the previous point
    slope = np.zeros(shape=(kdest.size-1, ))
    for idx in range(len(slope.size)):
        slope[idx] = kdest[idx+1] - kdest[idx]

    # Compute sign of slope points in order to find peaks and troughs
    signs = np.sign(slope)

    # Multiply consecutive signs to arrive at points which are peaks and troughs
    inflexions = np.zeros(shape=(signs.size-1, ))
    for idx in range(len(inflexions.size)):
        inflexions[idx] = signs[idx+1] * signs[idx]

    inf_idx = np.where(inflexions == -1)[0]

    # Store the inflexion points in a 2D array and return it
    inflexion_points = np.zeros(shape=(inf_idx.size, 2))
    inflexion_points[:, 0] = range_[inf_idx, ]
    inflexion_points[:, 1] = kdest[inf_idx, ]

    return inflexion_points


def categorize_parameter(data, num_values, inflexion_points=None):
    """
    :param data:
    :return:
    """

    # Find out Inflexion points if it is not provided
    if inflexion_points is None:
        inflexion_points = kde(data, num_values/1000)

