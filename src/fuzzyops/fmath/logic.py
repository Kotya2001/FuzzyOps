import numpy as np
from numba import cuda
import numba
from time import perf_counter
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


def in_cuda(func):

    def inner(v1, v2):
        v1_device = cuda.to_device(v1)
        v2_device = cuda.to_device(v2)
        st = perf_counter()
        res = func(v1_device, v2_device).copy_to_host()
        en = perf_counter()
        print('interf time', en - st)
        del v1_device
        del v2_device
        return res

    return inner


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


def fuzzy_and_mm(vals1, vals2):
    """Logical and of two FuzzyNumbers by minimax method.

    Parameters
    ----------
    vals1, vals2 : `numpy.ndarray`

    Returns
    -------
    values : `numpy.ndarray`
    """
    values = elementwise_min(vals1, vals2)

    return values


@in_cuda
def fuzzy_or_mm(vals1, vals2):
    """Logical or of two FuzzyNumbers by minimax method.

    Parameters
    ----------
    vals1, vals2 : `numpy.ndarray`

    Returns
    -------
    values : `numpy.ndarray`
    """
    values = elementwise_max(vals1, vals2)

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
    values = elementwise_mul(vals1, vals2)

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
    values = elementwise_or(vals1, vals2)

    return values
