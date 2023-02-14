import numpy as np


def triangularmf(x, a, b, c):
    """Triangular membership function

    Parameters
    ----------
    x : `numpy.ndarray`
        Range on which the membership is based on.
    a : `float`
        Left corner of the triangle. Value of x on which the membership is equal to 0.
    b : `float`
        Peak of the triangle. Value of x on which the membership is equal to 1.
    c : `float`
        Right corner of the triangle. Value of x on which the membership is equal to 0.

    Returns
    -------
    y : `numpy.ndarray`
        Membership function.
    """
    assert a <= b <= c, "a <= b <= c"
    y = np.zeros(len(x))
    if a != b:
        idx = np.argwhere((a < x) & (x < b))
        y[idx] = (x[idx] - a) / float(b - a)
    if b != c:
        idx = np.argwhere((b < x) & (x < c))
        y[idx] = (c - x[idx]) / float(c - b)
    idx = np.nonzero(x == b)
    y[idx] = 1
    return y


def trapezoidalmf(x, a, b, c, d):
    """Trapezoidal membership function

    Parameters
    ----------
    x : `numpy.ndarray`
        Range on which the membership is based on.
    a : `float`
        Left corner of the trapezia. Value of x on which the membership is equal to 0.
    b : `float`
        Left peak of the trapezia. Value of x on which the membership is equal to 1.
    c : `float`
        Right peak of the trapezia. Value of x on which the membership is equal to 1.
    d : `float`
        Right corner of the trapezia. Value of x on which the membership is equal to 0.

    Returns
    -------
    y : `numpy.ndarray`
        Membership function.
    """
    assert a <= b <= c, "a <= b <= c <= d"
    y = np.zeros(len(x))
    if a != b:
        idx = np.argwhere((a <= x) & (x <= b))
        y[idx] = (x[idx] - a) / float(b - a)
    idx = np.nonzero(np.logical_and(b < x, x < c))[0]
    y[idx] = 1
    if c != d:
        idx = np.argwhere((c <= x) & (x <= d))
        y[idx] = (d - x[idx]) / float(d - c)
    return y


def gaussmf(x, sigma, mean):
    y = np.exp(-np.power(x - mean, 2) / (2*sigma**2))
    return y

# TODO: more memberships


memberships = {'triangular': triangularmf, 'trapezoidal': trapezoidalmf, 'gauss': gaussmf}