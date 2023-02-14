import time

import numpy as np
from numba import cuda
import numba
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


@cuda.jit
def cuda_max(result: np.ndarray, v1: np.ndarray, v2: np.ndarray):
    i = cuda.grid(1)
    result[i] = max(v1[i], v2[i])


@cuda.jit
def cuda_or(result: np.ndarray, v1: np.ndarray, v2: np.ndarray):
    i = cuda.grid(1)
    result[i] = v1[i] + v2[i] - v1[i] * v2[i]


@cuda.jit
def cuda_min(result: np.ndarray, v1: np.ndarray, v2: np.ndarray):
    i = cuda.grid(1)
    result[i] = min(v1[i], v2[i])


@cuda.jit
def cuda_mul(result: np.ndarray, v1: np.ndarray, v2: np.ndarray):
    i = cuda.grid(1)
    result[i] = v1[i] * v2[i]


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
    else:
        result = np.zeros_like(vals1)
        threadsperblock = 1024
        blockspergrid = math.ceil(result.shape[0] / threadsperblock)
        cuda_min[blockspergrid, threadsperblock](result, vals1, vals2)
        return result



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
    else:
        result = np.zeros_like(vals1)
        threadsperblock = 1024
        blockspergrid = math.ceil(result.shape[0] / threadsperblock)
        cuda_max[blockspergrid, threadsperblock](result, vals1, vals2)
        return result


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
    else:
        result = np.zeros_like(vals1)
        threadsperblock = 1024
        blockspergrid = math.ceil(result.shape[0] / threadsperblock)
        cuda_mul[blockspergrid, threadsperblock](result, vals1, vals2)

        return result


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
    else:
        result = np.zeros_like(vals1)
        threadsperblock = 1024
        blockspergrid = math.ceil(result.shape[0] / threadsperblock)
        cuda_min[blockspergrid, threadsperblock](result, vals1, vals2)
        return result

