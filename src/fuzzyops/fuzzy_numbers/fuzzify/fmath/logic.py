
import torch
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
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.minimum(mf1(x), mf2(x))
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

    def f(x: torch.Tensor):
        return torch.maximum(mf1(x), mf2(x))

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

    def f(x: torch.Tensor):
        return mf1(x) * mf2(x)

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
    def f(x: torch.Tensor):
        vals1 = mf1(x)
        vals2 = mf2(x)
        return vals1 + vals2 - vals1 * vals2

    return f

