import numpy as np
import matplotlib.pyplot as plt
from .fmath.operations import fuzzy_difference, fuzzy_unite, fuzzy_intersect
from .fmath.logic import dtype
from .defuzz import DEFAULT_DEFUZZ
from .fuzzify.mf import very, neg, maybe
from typing import Union, Callable, List, Tuple

AnyNum = Union['FuzzyNumber', int, float]
RealNum = Union[int, float]


class FuzzyNumber(object):
    """Fuzzy Number.
    Set on a domain, membership is represented by np.array

    Parameters
    ----------
    domain : `fuzzyops.Domain`
        Domain on which the number is based.
    values : `numpy.ndarray`
        Values that represent membership of the fuzzy number.
    method : `str`
        Method of calculations: `minimax` or `prob`. Default is `minimax`.

    Methods
    -------

    """
    def __init__(self, domain, membership: Callable, method: str = 'minimax'):
        assert method == 'minimax' or method == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        self._domain = domain
        self._membership = membership
        self._values = membership(self._domain.x).astype(dtype)
        self._method = method

    @property
    def very(self):
        """Copy of the number with membership in a power of 2.
        """
        return FuzzyNumber(self._domain, very(self._membership), self._method)

    @property
    def negation(self):
        """Copy of the number with opposite membership.
        """
        return FuzzyNumber(self._domain, neg(self._membership), self._method)

    @property
    def maybe(self):
        """Copy of the number with membership in a power of 0.5.
        """
        return FuzzyNumber(self._domain, maybe(self._membership), self._method)

    def copy(self):
        return FuzzyNumber(self._domain, self._membership, self._method)

    @property
    def method(self):
        return self._method

    @property
    def membership(self):
        return self._membership

    @property
    def domain(self):
        return self._domain

    @property
    def values(self):
        return self._values

    def plot(self, ax=None):
        """Plots the number. Creates new subplot if not specified.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        if not ax:
            _, ax = plt.subplots()
        out = ax.plot(self.domain.x, self._values)
        plt.show()
        return out

    def alpha_cut(self, alpha:float):
        """Alpha-cut of a number.

        Parameters
        ----------
        alpha : `float`

        Returns
        -------
        value : `numpy.ndarray`
        """
        return self.domain.x[self._values >= alpha]

    def entropy(self, norm: bool = True):
        e = -np.sum(self.values * np.log2(self.values, out=np.zeros_like(self.values), where=(self.values != 0)))
        if norm:
            return 2./len(self.values) * e
        else:
            return e

    def center_of_grav(self):
        return np.sum(self.domain.x*self.values) / np.sum(self.values)

    def left_max(self):
        h = np.max(self.values)
        return self.domain.x[self.values == h][0]

    def right_max(self):
        h = np.max(self.values)
        return self.domain.x[self.values == h][1]

    def center_of_max(self, verbose: bool = False):
        h = np.max(self.values)
        maxs = self.domain.x[self.values == h]
        if verbose:
            print('h:', h, 'maximums are:', maxs)
        return np.mean(maxs)

    def moment_of_inertia(self, center: bool = None):
        if not center:
            center = self.center_of_grav()
        return np.sum(self.values * np.square(self.domain.x - center))

    def defuzz(self, by: str = 'default'):
        if by == 'default':
            by = DEFAULT_DEFUZZ
        if by == 'lmax':
            return self.left_max()
        elif by == 'rmax':
            return self.right_max()
        elif by == 'cmax':
            return self.center_of_max()
        elif by == 'cgrav':
            return self.center_of_grav()
        else:
            raise ValueError('defuzzification can be made by lmax, rmax, cmax, cgrav of default')

    # magic
    def __str__(self):
        return str(self.defuzz())

    def __repr__(self):
        return 'Fuzzy' + str(self.defuzz())

    def __add__(self, other: AnyNum):
        if isinstance(other, int) or isinstance(other, float):
            def added(x):
                return self._membership(x-other)
            return FuzzyNumber(self.domain, added, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_unite(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only add a number (Fuzzynumber, int or float)')

    def __iadd__(self, other: AnyNum):
        return self + other

    def __sub__(self, other: AnyNum):
        if isinstance(other, int) or isinstance(other, float):
            def diff(x):
                return self._membership(x - other)

            return FuzzyNumber(self.domain, diff, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_difference(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only substract a number (Fuzzynumber, int or float)')

    def __isub__(self, other: AnyNum):
        return self - other

    def __mul__(self, other: AnyNum):  # change
        if isinstance(other, int) or isinstance(other, float):
            def multiplied(x):
                return self._membership(x * other)

            return FuzzyNumber(self.domain, multiplied, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_intersect(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only substract a number (Fuzzynumber, int or float)')

    def __imul__(self, other: AnyNum):
        return self * other

    def __truediv__(self, other: RealNum):
        t_o = type(other)
        if t_o == int or t_o == float:
            def divided(x):
                return self._membership(x / other)

            return FuzzyNumber(self.domain, divided, self._method)
        else:
            raise TypeError('can only divide by a number (int or float)')

    def __idiv__(self, other: RealNum):
        return self / other

    def __int__(self):
        return int(self.defuzz())

    def __float__(self):
        return self.defuzz()
