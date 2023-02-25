import numpy as np
from typing import Callable, Union


def very(mf: Callable) -> Callable:
    def f(x):
        return mf(x)**2
    return f


def maybe(mf: Callable) -> Callable:
    def f(x):
        return mf(x)**0.5
    return f


def neg(mf: Callable) -> Callable:
    def f(x):
        return 1-mf(x)
    return f


def triangularmf(a: Union[int, float], b: Union[int, float], c: Union[int, float]) -> Callable:
    """Triangular membership function

    Parameters
    ----------
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

    def f(x) -> np.ndarray:
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
    return f


def trapezoidalmf(a: Union[int, float], b: Union[int, float], c: Union[int, float], d: Union[int, float]) -> Callable:
    """Trapezoidal membership function

    Parameters
    ----------
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

    def f(x: np.ndarray) -> np.ndarray:
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
    return f


def gaussmf(sigma: Union[int, float], mean: Union[int, float]) -> Callable:
    """

    Parameters
    ----------
    sigma : `int` or `float`
    mean : `int` or `float`

    Returns
    -------
    mf : Callable
    """
    def f(x: np.ndarray) -> np.ndarray:
        y = np.exp(-np.power(x - mean, 2) / (2*sigma**2))
        return y
    return f

# TODO: more memberships


memberships = {'triangular': triangularmf, 'trapezoidal': trapezoidalmf, 'gauss': gaussmf}
