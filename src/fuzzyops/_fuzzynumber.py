import numpy as np
import matplotlib.pyplot as plt
from .math import fuzzy_difference, fuzzy_unite, fuzzy_intersect
from .defuzz import DEFAULT_DEFUZZ


class FuzzyNumber(object):
    """Fuzzy Number.
    Consists of two arrays: set and values

    Parameters
    ----------
    x : `numpy.ndarray`
        Set on which the number is based.
    values : `numpy.ndarray`
        Values that represent membership of the fuzzy number.
    method : `str`
        Method of calculations: `minimax` or `prob`. Default is `minimax`.

    Methods
    -------

    """
    def __init__(self, x, values, method='minimax'):
        assert method == 'minimax' or method == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        self._x = x
        self._values = values
        self._method = method

    @property
    def very(self):
        """Copy of the number with membership in a power of 2.
        """
        return FuzzyNumber(self.get_x(), np.power(self.get_values(), 2))

    @property
    def negation(self):
        """Copy of the number with opposite membership.
        """
        return FuzzyNumber(self.get_x(), 1.-self.get_values())

    @property
    def maybe(self):
        """Copy of the number with membership in a power of 0.5.
        """
        return FuzzyNumber(self.get_x(), np.power(self.get_values(), 0.5))

    def get_method(self):
        return self._method

    def get_x(self):
        return self._x

    def get_values(self):
        return self._values

    def plot(self, ax=None):
        """Plots the number. Creates new subplot if none is specified.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        if not ax:
            _, ax = plt.subplots()
        ax.plot(self.get_x(), self.get_values())

        ax.set(ylim=(0, 1), yticks=np.arange(0, 1, 0.1),
               xticks=np.arange(self._x[0], self._x[-1], 0.5))

        plt.show()

    def extend_values(self, fset, inplace=False):
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
        #  return new fnum shaped to fit given range
        assert fset[0] <= self.get_x()[0] and fset[-1] >= self.get_x()[-1], "New set must include existing range"

        if not inplace:
            clone = FuzzyNumber(self.get_x(), self.get_values(), self.get_method())
            clone._values = np.interp(fset, self.get_x(), self.get_values())
            clone._x = fset
            return clone
        else:
            self._values = np.interp(fset, self._x, self._values)
            self._x = fset

    def alpha_cut(self, alpha):
        """Alpha-cut of a number.

        Parameters
        ----------
        alpha : `float`

        Returns
        -------
        value : `numpy.ndarray`
        """
        return self.get_x()[self._values >= alpha]

    def entropy(self, norm=True):
        e = -np.sum(self.get_values() * np.log2(self.get_values(), out=np.zeros_like(self.get_values()), where=(self.get_values()!=0)))
        if norm:
            return 2./len(self.get_values()) * e
        else:
            return e

    def center_of_grav(self):
        return np.sum(self.get_x()*self.get_values()) / np.sum(self.get_values())

    def left_max(self):
        h = np.max(self.get_values())
        return self.get_x()[self.get_values() == h][0]

    def right_max(self):
        h = np.max(self.get_values())
        return self.get_x()[self.get_values() == h][1]

    def center_of_max(self, verbose=False):
        h = np.max(self.get_values())
        maxs = self.get_x()[self.get_values() == h]
        if verbose:
            print('h:', h, 'maximums are:', maxs)
        return np.mean(maxs)

    def moment_of_inertia(self, center=None):
        if not center:
            center = self.center_of_grav()
        return np.sum(self.get_values()*np.square(self.get_x()-center))

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

    def __add__(self, other):
        if isinstance(other, FuzzyNumber):
            xs, vals = fuzzy_unite(self, other)
            return FuzzyNumber(xs, vals, self._method)
        elif isinstance(other, int) or isinstance(other, float):
            return FuzzyNumber(self.get_x() + other, self.get_values(), self.get_method())
        else:
            raise TypeError('can only add a number (Fuzzynumber, int or float)')

    def __iadd__(self, other):
        new = self + other
        self._x = new.get_x()

    def __sub__(self, other):
        if isinstance(other, FuzzyNumber):
            xs, vals = fuzzy_difference(self, other)
            return FuzzyNumber(xs, vals, self._method)
        elif isinstance(other, int) or isinstance(other, float):
            return FuzzyNumber(self.get_x() - other, self.get_values(), self.get_method())
        else:
            raise TypeError('can only substract a number (Fuzzynumber, int or float)')

    def __isub__(self, other):
        new = self - other
        self._x = new.get_x()

    def __mul__(self, other):
        t_o = type(other)
        if t_o == FuzzyNumber:
            xs, vals = fuzzy_intersect(self, other)
            return FuzzyNumber(xs, vals, self.get_method())
        elif t_o == int or t_o == float:
            return FuzzyNumber(self.get_x()*other, self.get_values(), self.get_method())
        else:
            raise TypeError('can only multiply by a number (Fuzzynumber, int or float)')

    def __imul__(self, other):
        new = self * other
        self._x = new.get_x()

    def __div__(self, other):
        t_o = type(other)
        if t_o == int or t_o == float:
            return FuzzyNumber(self.get_x() / other, self.get_values(), self.get_method())
        else:
            raise TypeError('can only divide by a number (int or float)')

    # TODO: proper division magic
    def __floordiv__(self, other):
        pass

    def __idiv__(self, other):
        new = self / other
        self._x = new.get_x()

    def __ifloordiv__(self, other):
        pass

    def __int__(self, other):
        return int(self.defuzz())

    def __float__(self, other):
        return self.defuzz()