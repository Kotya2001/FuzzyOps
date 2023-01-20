import numpy as np
from numba import cuda
import numba
dtype = 'float32'


def unite_fsets_old(fnum1, fnum2):
    """Returns an x appropriate for both FuzzyNumbers

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    X : `numpy.ndarray`
    """
    mins = (fnum1.get_x()[0], fnum2.get_x()[0])
    steps = (fnum1.get_x()[1] - fnum1.get_x()[0], fnum2.get_x()[1] - fnum2.get_x()[0])
    maxs = (fnum1.get_x()[-1], fnum2.get_x()[-1])
    mi = np.min(mins)
    ma = np.max(maxs)
    step = np.min(steps)
    X = np.arange(mi, ma + step, step)

    return X


#@numba.njit
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


@numba.vectorize([f'{dtype}({dtype}, {dtype})'], target='cuda')
def elementwise_max(v1, v2):
    return max(v1, v2)


@numba.vectorize([f'{dtype}({dtype}, {dtype})'], target='cuda')
def elementwise_min(v1, v2):
    return min(v1, v2)


@numba.vectorize([f'{dtype}({dtype}, {dtype})'], target='cuda')
def elementwise_mul(v1, v2):
    return v1 * v2


@numba.vectorize([f'{dtype}({dtype}, {dtype})'], target='cuda')
def elementwise_or(v1, v2):
    return v1 + v2 - v1 * v2


def fuzzy_and_mm(fnum1, fnum2):
    """Logical and of two FuzzyNumbers by minimax method.

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    xs : `numpy.ndarray`
    values : `numpy.ndarray`
    """
    if np.array_equal(fnum1.get_x(), fnum2.get_x()):
        xs = fnum1.get_x()
    else:
        #xs = unite_fsets(fnum1, fnum2)
        xs = np.array(unite_fsets(fnum1.get_x(), fnum2.get_x()))
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = elementwise_min(fnum1.get_values(), fnum2.get_values())

    return xs, values


def fuzzy_or_mm(fnum1, fnum2):
    """Logical or of two FuzzyNumbers by minimax method.

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    xs : `numpy.ndarray`
    values : `numpy.ndarray`
    """
    if np.array_equal(fnum1.get_x(), fnum2.get_x()):
        xs = fnum1.get_x()
    else:
        xs = np.array(unite_fsets(fnum1.get_x(), fnum2.get_x()))
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = elementwise_max(fnum1.get_values().astype(dtype), fnum2.get_values().astype(dtype))

    return xs, values


def fuzzy_and_prob(fnum1, fnum2):
    """Logical and of two FuzzyNumbers by probabilistic method.

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    xs : `numpy.ndarray`
    values : `numpy.ndarray`
    """
    if np.array_equal(fnum1.get_x(), fnum2.get_x()):
        xs = fnum1.get_x()
    else:
        xs = np.array(unite_fsets(fnum1.get_x(), fnum2.get_x()))
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = elementwise_mul(fnum1.get_values(), fnum2.get_values())

    return xs, values


def fuzzy_or_prob(fnum1, fnum2):
    """Logical or of two FuzzyNumbers by probabilistic method.

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    xs : `numpy.ndarray`
    values : `numpy.ndarray`
    """
    if np.array_equal(fnum1.get_x(), fnum2.get_x()):
        xs = fnum1.get_x()
    else:
        xs = np.array(unite_fsets(fnum1.get_x(), fnum2.get_x()))
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = elementwise_or(fnum1.get_values(), fnum2.get_values())

    return xs, values
