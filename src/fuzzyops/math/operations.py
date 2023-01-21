import numpy as np
from .logic import fuzzy_or_prob, fuzzy_or_mm, \
    fuzzy_and_prob, fuzzy_and_mm, unite_fsets


def fuzzy_unite(fnum1, fnum2):
    """Returns a union of values of two FuzzyNumbers

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    value : `numpy.ndarray`
    """
    if fnum1._method == 'prob':
        return fuzzy_or_prob(fnum1, fnum2)
    elif fnum1._method == 'minimax':
        return fuzzy_or_mm(fnum1, fnum2)
    else:
        raise ValueError('Only minimax and prob methods are supported')


def fuzzy_intersect(fnum1, fnum2):
    """Returns an intersection of values of two FuzzyNumbers

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    value : `numpy.ndarray`
    """
    if fnum1._method == 'prob':
        return fuzzy_and_prob(fnum1, fnum2)
    elif fnum1._method == 'minimax':
        return fuzzy_and_mm(fnum1, fnum2)
    else:
        raise ValueError('Only minimax and prob methods are supported')


def fuzzy_difference(fnum1, fnum2):
    """Returns a difference of values of two FuzzyNumbers

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    value : `numpy.ndarray`
    """
    if np.array_equal(fnum1._x, fnum2._x):
        xs = fnum1._x
    else:
        xs = unite_fsets(fnum1, fnum2)
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = np.clip(fnum1._values - fnum2._values, 0, 1)

    return xs, values
