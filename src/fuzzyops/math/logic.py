import numpy as np


def unite_fsets(fnum1, fnum2):
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
        xs = unite_fsets(fnum1, fnum2)
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = np.minimum(fnum1.get_values(), fnum2.get_values())

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
        xs = unite_fsets(fnum1, fnum2)
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = np.maximum(fnum1.get_values(), fnum2.get_values())

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
        xs = unite_fsets(fnum1, fnum2)
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = fnum1.get_values() * fnum2.get_values()

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
        xs = unite_fsets(fnum1, fnum2)
        fnum1 = fnum1.extend_values(xs)
        fnum2 = fnum2.extend_values(xs)
    values = fnum1.get_values() + fnum2.get_values() - fnum1.get_values() * fnum2.get_values()

    return xs, values
