import torch
from typing import Callable, Tuple
from .logic import fuzzy_or_prob, fuzzy_or_mm, \
    fuzzy_and_prob, fuzzy_and_mm


def fuzzy_unite(fnum1, fnum2) -> Tuple:
    """Returns a union of values of two FuzzyNumbers

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    membership_tuple : `Tuple`
    """

    if fnum1._method == 'prob':
        return fuzzy_or_prob(fnum1.membership, fnum2.membership)
    elif fnum1._method == 'minimax':
        return fuzzy_or_mm(fnum1.membership, fnum2.membership)
    else:
        raise ValueError('Only minimax and prob methods are supported')


def fuzzy_intersect(fnum1, fnum2) -> Tuple:
    """Returns an intersection of values of two FuzzyNumbers

    Parameters
    ----------
    fnum1, fnum2 : `FuzzyNumber`

    Returns
    -------
    membership_tuple : `Tuple`
    """
    if fnum1._method == 'prob':
        return fuzzy_and_prob(fnum1.membership, fnum2.membership)
    elif fnum1._method == 'minimax':
        return fuzzy_and_mm(fnum1.membership, fnum2.membership)
    else:
        raise ValueError('Only minimax and prob methods are supported')


def fuzzy_difference(fnum1, fnum2):
    def f(vals1: torch.Tensor, vals2: torch.Tensor):
        return torch.clip(vals1 - vals2, 0, 1)
    return (f, fnum1.membership, fnum2.membership)
    
# def fuzzy_difference(fnum1, fnum2) -> Callable:
#     """Returns a difference of values of two FuzzyNumbers

#     Parameters
#     ----------
#     fnum1, fnum2 : `FuzzyNumber`

#     Returns
#     -------
#     function : `Callable`
#     """

#     def f(x):
#         dx = fnum1.domain.x
#         values = torch.clip(fnum1.membership(dx) - fnum2.membership(dx), 0, 1)
#         return values

#     return f
