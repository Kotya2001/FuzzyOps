import time

import numpy as np
from typing import Tuple, Callable
dtype = 'float32'


def fuzzy_and_mm(mf1: Callable, mf2: Callable) -> Callable:
    """Logical and of two FuzzyNumbers by minimax method.

    Parameters
    ----------
    mf1, mf2 : `Callable`

    Returns
    -------
    function : `Callable`
    """
    def f(x: np.ndarray):
        print(mf1)
        print(mf2)
        return np.minimum(mf1(x), mf2(x))
    return f


def fuzzy_or_mm(mf1: Callable, mf2: Callable) -> Callable:
    """Logical or of two FuzzyNumbers by minimax method.

    Parameters
    ----------
    mf1, mf2 : `Callable`

    Returns
    -------
    function : `Callable`
    """

    def f(x: np.ndarray):
        return np.maximum(mf1(x), mf2(x))

    return f


def fuzzy_and_prob(mf1: Callable, mf2: Callable) -> Callable:
    """Logical and of two FuzzyNumbers by probabilistic method.

    Parameters
    ----------
    mf1, mf2 : `Callable`

    Returns
    -------
    function : `Callable`
    """

    def f(x: np.ndarray):
        return np.multiply(mf1(x), mf2(x))

    return f


def fuzzy_or_prob(mf1: Callable, mf2: Callable) -> Callable:
    """Logical or of two FuzzyNumbers by probabilistic method.

    Parameters
    ----------
    mf1, mf2 : `Callable`

    Returns
    -------
    function : `Callable`
    """
    def f(x: np.ndarray):
        vals1 = mf1(x)
        vals2 = mf2(x)
        return vals1 + vals2 - np.multiply(vals1, vals2)

    return f

