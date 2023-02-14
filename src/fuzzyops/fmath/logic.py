import time

import numpy as np
from time import perf_counter
import math
dtype = 'float32'


def unite_fsets(x1, x2):
    """Returns an x appropriate for both FuzzyNumbers

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    X : `numpy.ndarray`
    """
    mins = (x1[0], x2[0])
    steps = (x1[1] - x1[0], x2[1] - x2[0])
    maxs = (x1[-1], x2[-1])
    mi = np.min(mins)
    ma = np.max(maxs)
    step = np.min(steps)
    X = np.arange(mi, ma + step, step)

    return X


def fuzzy_and_mm(vals1, vals2):
    """Logical and of two FuzzyNumbers by minimax method.

    Parameters
    ----------
    vals1, vals2 : `numpy.ndarray`

    Returns
    -------
    values : `numpy.ndarray`
    """

    if isinstance(vals1, np.ndarray):
        values = np.minimum(vals1, vals2)
        return values



def fuzzy_or_mm(vals1, vals2):
    """Logical or of two FuzzyNumbers by minimax method.

    Parameters
    ----------
    vals1, vals2 : `numpy.ndarray`

    Returns
    -------
    values : `numpy.ndarray`
    """
    if isinstance(vals1, np.ndarray):
        values = np.maximum(vals1, vals2)
        return values


def fuzzy_and_prob(vals1, vals2):
    """Logical and of two FuzzyNumbers by probabilistic method.

    Parameters
    ----------
    vals1, vals2 : `numpy.ndarray`

    Returns
    -------
    values : `numpy.ndarray`
    """

    if isinstance(vals1, np.ndarray):
        values = np.multiply(vals1, vals2)
        return values


def fuzzy_or_prob(vals1, vals2):
    """Logical or of two FuzzyNumbers by probabilistic method.

    Parameters
    ----------
    vals1, vals2 : `numpy.ndarray`

    Returns
    -------
    values : `numpy.ndarray`
    """
    if isinstance(vals1, np.ndarray):
        values = vals1 + vals2 - np.multiply(vals1, vals2)
        return values

