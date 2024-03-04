import torch
from typing import Callable, Union


def very(mf: Callable) -> Callable:
    def f(x):
        return mf(x) ** 2

    return f


def maybe(mf: Callable) -> Callable:
    def f(x):
        return mf(x) ** 0.5

    return f


def neg(mf: Callable) -> Callable:
    def f(x):
        return 1 - mf(x)

    return f


def clip_upper(mf: Callable, upper: Union[int, float]) -> Callable:
    def f(x):
        return torch.minimum(mf(x), torch.tensor([upper]))

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

    def f(x) -> torch.Tensor:
        y = torch.zeros(len(x))
        if a != b:
            idx = torch.argwhere((a < x) & (x < b))
            y[idx] = (x[idx] - a) / float(b - a)
        if b != c:
            idx = torch.argwhere((b < x) & (x < c))
            y[idx] = (c - x[idx]) / float(c - b)
        idx = torch.nonzero(x == b)
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

    def f(x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(len(x)).to(x.device)
        if a != b:
            idx = torch.argwhere((a <= x) & (x <= b))
            y[idx] = (x[idx] - a) / float(b - a)
        idx = torch.nonzero(torch.logical_and(b < x, x < c))
        y[idx] = 1
        if c != d:
            idx = torch.argwhere((c <= x) & (x <= d))
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

    def f(x: torch.Tensor) -> torch.Tensor:
        y = torch.exp(-torch.pow(x - mean, 2) / (2 * sigma ** 2))
        return y

    return f


def generalized_bell_mf(a: Union[int, float], b: Union[int, float], c: Union[int, float]) -> Callable:
    """

    Parameters
    ----------
    a : `int` or `float`
    b : `int` or `float`
    c : `int` or `float`

    Returns
    -------
    mf : Callable
    """

    def f(x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.pow(torch.abs((x - c) / a), 2 * b))
    return f


# TODO: more memberships


memberships = {'triangular': triangularmf, 'trapezoidal': trapezoidalmf,
               'gauss': gaussmf, 'bell': generalized_bell_mf}
