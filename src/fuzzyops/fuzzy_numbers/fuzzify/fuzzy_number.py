from .fmath import fuzzy_unite, fuzzy_difference, fuzzy_intersect
from .mf import very, neg, maybe, memberships
from .defuzz import DEFAULT_DEFUZZ
from typing import Callable, Union, Tuple

import torch
import matplotlib.pyplot as plt
from inspect import signature

RealNum = Union[float, int]
AnyNum = Union['FuzzyNumber', int, float]

default_dtype = "float32"


class Domain:
    """
    Domain that represents a set of possible values of a number
    """

    def __init__(self, fset: Union[Tuple[RealNum, RealNum], Tuple[RealNum, RealNum, RealNum], torch.Tensor],
                 name: str = None, method: str = 'minimax'):
        assert (isinstance(fset, Tuple) and (len(fset) == 3 or len(fset) == 2)) or isinstance(fset, torch.Tensor), \
            'set bust be given as torch.Tensor or tuple with start, end, step values'
        assert method == 'minimax' or method == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        if isinstance(fset, torch.Tensor):
            self._x = fset
            self.step = self._x[1] - self._x[0]
        elif len(fset) == 3:
            start, end, step = fset
            self._x = torch.arange(start, end, step)
            self.step = step
        elif len(fset) == 2:
            start, end = fset
            self._x = torch.arange(start, end, 1)
            self.step = 1
        self.name = name
        self._method = method
        self._vars = {}

    @property
    def method(self):
        """Returns the method used for fuzzy operations"""
        return self._method

    @method.setter
    def method(self, value):
        assert value == 'minimax' or value == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        self._method = value
        for name, var in self._vars.items():
            var._method = value

    def to(self, device: str):
        """Moves domain to device"""
        self._x = self._x.to(device)

    @property
    def x(self):
        """Returns the domain range"""
        return self._x

    def create_number(self, membership: Union[str, Callable], *args: RealNum, name: str = None):
        """Creates new FuzzyNumber in the domain"""
        assert isinstance(membership, str) or (isinstance(membership, Callable) and
                                               len(args) == len(signature(membership).parameters))
        if isinstance(membership, str):
            membership = memberships[membership]
        f = FuzzyNumber(self, membership(*args), self._method)
        if name:
            self.__setattr__(name, f)
        return f

    def __setattr__(self, name, value):
        if name in ['_x', 'step', 'name', '_method', '_vars', 'method']:
            object.__setattr__(self, name, value)
        else:
            # assert isinstance(name, str) and name not in self._vars, 'Name must be a unique string'
            assert isinstance(value, FuzzyNumber), 'Value must be FuzzyNumber'
            self._vars[name] = value

    def __getattr__(self, name):
        if name in self._vars:
            return self._vars[name]
        else:
            raise AttributeError(f'{name} is not a variable in domain {self.name}')

    def get(self, name):
        """Returns FuzzyNumber with given name"""
        return self._vars[name]

    def __delattr__(self, name):
        if name in self._vars:
            del self._vars[name]

    def plot(self):
        """Plots all variables in the domain"""
        _, ax = plt.subplots()

        for name, num in self._vars.items():
            ax.plot(self.x, num.values, label=f'{name}')

        plt.title(self.name)
        ax.legend()
        plt.show()


class FuzzyNumber:
    """Fuzzy Number.
    Set on a domain, membership is represented by np.array
    Parameters
    ----------
    domain : `fuzzyops.Domain`
        Domain on which the number is based.
    values : `torch.Tensor`
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
        # self._values = membership(self._domain.x).astype(default_dtype)
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
    def values(self, dtype: str = default_dtype):
        return self.membership(self._domain.x)  # .astype(dtype)

    def plot(self, ax=None):
        """Plots the number. Creates new subplot if not specified.
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        if self.domain.x.device != 'cpu':
            raise TypeError("can't convert cuda:0 device type tensor to numpy. Use Domain.to('cpu') first.")
        if not ax:
            _, ax = plt.subplots()
        out = ax.plot(self.domain.x, self.values)
        plt.show()
        return out

    def alpha_cut(self, alpha: float):
        """Alpha-cut of a number.
        Parameters
        ----------
        alpha : `float`
        Returns
        -------
        value : `numpy.ndarray`
        """
        return self.domain.x[self.values >= alpha]

    def entropy(self, norm: bool = True):
        """Entropy of the number.
        Parameters
        ----------
        norm : `bool`
            If True, entropy is normalized by the number of elements in the domain.
        Returns
        -------
        entropy : `float`
        """
        vals = self.values
        mask = vals != 0
        e = -torch.sum(vals[mask] * torch.log2(vals[mask]))
        if norm:
            return 2. / len(self.values) * e
        else:
            return e

    def center_of_grav(self):
        return float(torch.sum(self.domain.x * self.values) / torch.sum(self.values))

    def left_max(self):
        h = torch.max(self.values)
        return float(self.domain.x[self.values == h][0])

    def right_max(self):
        h = torch.max(self.values)
        return float(self.domain.x[self.values == h][1])

    def center_of_max(self, verbose: bool = False):
        h = torch.max(self.values)
        maxs = self.domain.x[self.values == h]
        if verbose:
            print('h:', h, 'maximums are:', maxs)
        return float(torch.mean(maxs))

    def moment_of_inertia(self, center: bool = None):
        if not center:
            center = self.center_of_grav()
        return float(torch.sum(self.values * torch.square(self.domain.x - center)))

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
            raise ValueError('defuzzification can be made by lmax, rmax, cmax, cgrav or default')

    # magic
    def __str__(self):
        return str(self.defuzz())

    def __repr__(self):
        return 'Fuzzy' + str(self.defuzz())

    def __add__(self, other: AnyNum):
        if isinstance(other, int) or isinstance(other, float):
            def added(x):
                return self._membership(x - other)

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
                return self._membership(x + other)

            return FuzzyNumber(self.domain, diff, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_difference(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only substract a number (Fuzzynumber, int or float)')

    def __isub__(self, other: AnyNum):
        return self - other

    def __mul__(self, other: AnyNum):
        if isinstance(other, int) or isinstance(other, float):
            raise NotImplementedError('Multiplication by a number is not implemented yet')

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
        raise NotImplementedError('Division is not implemented yet')
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
