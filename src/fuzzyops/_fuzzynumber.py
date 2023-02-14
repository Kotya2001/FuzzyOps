import numpy as np
import matplotlib.pyplot as plt
from .fmath.operations import fuzzy_difference, fuzzy_unite, fuzzy_intersect
from .fmath.logic import dtype
from .defuzz import DEFAULT_DEFUZZ
#from ._domain import Domain
from typing import Union


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
    def __init__(self, domain, values: np.ndarray, method: str = 'minimax'):
        assert method == 'minimax' or method == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        self._domain = domain
        self._values = values.astype(dtype)
        self._method = method

    @property
    def very(self):
        """Copy of the number with membership in a power of 2.
        """
        return FuzzyNumber(self._domain, np.power(self.values, 2))

    @property
    def negation(self):
        """Copy of the number with opposite membership.
        """
        return FuzzyNumber(self._domain, 1.-self.values)

    @property
    def maybe(self):
        """Copy of the number with membership in a power of 0.5.
        """
        return FuzzyNumber(self._domain, np.power(self.values, 0.5))

    def copy(self):
        return FuzzyNumber(self._domain, self.values, self._method)

    def get_method(self):
        return self._method

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

        #ax.set(ylim=(0, 1), yticks=np.arange(0, 1, 0.1),
        #       xticks=np.arange(self._x[0], self._x[-1], 0.5))

        plt.show()
        return out

    def __extend_values(self, fset, inplace=False):
        """ Returns new FuzzyNumber shaped to fit new x values

        Parameters
        ----------
        fset : `np.ndarray`
            Desired set of a number.
        inplace : `bool`
            `False` if new number must be returned.

        Returns
        -------
        clone : `FuzzyNumber`
            Returned only if inplace=`False`.
        """
        assert fset[0] <= self.get_x()[0] and fset[-1] >= self.get_x()[-1], "New set must include existing range"

        if not inplace:
            clone = FuzzyNumber(self.get_x(), self.get_values(), self.get_method())
            clone._values = np.interp(fset, self.get_x(), self.get_values()).astype(dtype)
            clone._x = fset.astype(dtype)
            return clone
        else:
            self._values = np.interp(fset, self._x, self._values).astype(dtype)
            self._x = fset.astype(dtype)

    def alpha_cut(self, alpha):
        """Alpha-cut of a number.

        Parameters
        ----------
        alpha : `float`

        Returns
        -------
        value : `numpy.ndarray`
        """
        return self.domain.x[self._values >= alpha]

    def entropy(self, norm=True):
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

    def center_of_max(self, verbose=False):
        h = np.max(self.values)
        maxs = self.domain.x[self.values == h]
        if verbose:
            print('h:', h, 'maximums are:', maxs)
        return np.mean(maxs)

    def moment_of_inertia(self, center=None):
        if not center:
            center = self.center_of_grav()
        return np.sum(self.values * np.square(self.domain.x - center))

    def defuzz(self, by='default'):
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

    def __add__(self, other):  # change
        if isinstance(other, FuzzyNumber):
            vals = fuzzy_unite(self, other)
            return FuzzyNumber(self.domain, vals, self._method)
        elif isinstance(other, int) or isinstance(other, float):
            n_steps = int(other / self.domain.step)
            values = np.concatenate((np.full(n_steps, self.values[0]), self._values[:-n_steps]))
            return FuzzyNumber(self.domain, values, self._method)
        else:
            raise TypeError('can only add a number (Fuzzynumber, int or float)')

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):  # change
        if isinstance(other, FuzzyNumber):
            vals = fuzzy_difference(self, other)
            return FuzzyNumber(self.domain, vals, self._method)
        elif isinstance(other, int) or isinstance(other, float):
            n_steps = int(other / self.domain.step)
            values = np.concatenate((self._values[n_steps:], np.full(n_steps, self.values[-1])))
            return FuzzyNumber(self.domain, values, self._method)
        else:
            raise TypeError('can only substract a number (Fuzzynumber, int or float)')

    def __isub__(self, other):
        return self - other

    def __mul__(self, other):  # change
        t_o = type(other)
        if t_o == FuzzyNumber:
            vals = fuzzy_intersect(self, other)
            return FuzzyNumber(self.domain, vals, self._method)
        elif t_o == int or t_o == float:
            values = self.values
            return FuzzyNumber(self.domain, values, self.get_method())
        else:
            raise TypeError('can only multiply by a number (Fuzzynumber, int or float)')

    def __imul__(self, other):
        return self * other

    def __div__(self, other):  # change
        t_o = type(other)
        if t_o == int or t_o == float:
            return FuzzyNumber(self.domain, self.values, self.get_method())
        else:
            raise TypeError('can only divide by a number (int or float)')

    # TODO: proper division magic
    def __truediv__(self, other):
        pass

    def __idiv__(self, other):
        return self / other

    def __int__(self, other):
        return int(self.defuzz())

    def __float__(self, other):
        return self.defuzz()
