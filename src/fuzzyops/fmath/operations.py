import numpy as np
from .logic import fuzzy_or_prob, fuzzy_or_mm, \
    fuzzy_and_prob, fuzzy_and_mm, unite_fsets


def check_cuda(*vals):
    if vals[0].cuda:
        for i in range(1, len(vals)):
            vals[i].to_device()
    else:
        for i in range(1, len(vals)):
            vals[i].to_host()


def fuzzy_unite(fnum1, fnum2):
    """Returns a union of values of two FuzzyNumbers

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    value : `numpy.ndarray`
    """
    check_cuda(fnum1, fnum2)

    if fnum1._method == 'prob':
        return fuzzy_or_prob(fnum1.values, fnum2.values)
    elif fnum1._method == 'minimax':
        return fuzzy_or_mm(fnum1.values, fnum2.values)
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
        return fuzzy_and_prob(fnum1.values, fnum2.values)
    elif fnum1._method == 'minimax':
        return fuzzy_and_mm(fnum1.values, fnum2.values)
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
    values = np.clip(fnum1.values - fnum2.values, 0, 1)

    return values
