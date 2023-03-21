import numpy as np
import matplotlib.pyplot as plt
from ._fuzzynumber import FuzzyNumber
from typing import Callable, Union, Tuple, List
from .fuzzify.mf import memberships
from inspect import signature

RealNum = Union[float, int]


class Domain:
    """Domain that represents a set of possible values of a number

    """
    def __init__(self, fset: Union[Tuple[RealNum, RealNum], Tuple[RealNum, RealNum, RealNum], np.ndarray],
                 name: str = None, method: str = 'minimax'):
        assert (isinstance(fset, Tuple) and (len(fset) == 3 or len(fset) == 2)) or isinstance(fset, np.ndarray), \
            'set bust be given as np.array or tuple with start, end, step values'
        assert method == 'minimax' or method == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        if len(fset) == 3:
            start, end, step = fset
            self._x = np.arange(start, end, step)
            self.step = step
        elif len(fset) == 2:
            start, end = fset
            self._x = np.arange(start, end, 1)
            self.step = 1
        else:
            self._x = np.array(fset)
            self.step = self._x[1] - self._x[0]
        self.name = name
        self._method = method
        self._vars = {}

    @property
    def x(self):
        return self._x

    def create_number(self, membership: Union[str, Callable], *args: RealNum, name: str = None):
        assert isinstance(membership, str) or (isinstance(membership, Callable) and
                                               len(args) == len(signature(membership).parameters))
        if isinstance(membership, str):
            membership = memberships[membership]
        f = FuzzyNumber(self, membership(*args), self._method)
        if name:
            self.__setattr__(name, f)
        return f

    def __setattr__(self, name, value):
        if name in ['_x', 'step', 'name', '_method', '_vars']:
            object.__setattr__(self, name, value)
        else:
            assert isinstance(name, str) and name not in self._vars, 'Name must be a unique string'
            assert isinstance(value, FuzzyNumber), 'Value must be FuzzyNumber'
            self._vars[name] = value

    def __getattr__(self, name):
        if name in self._vars:
            return self._vars[name]
        else:
            raise AttributeError(f'{name} is not a variable in domain {self.name}')

    def __delattr__(self, name):
        if name in self._vars:
            del self._vars[name]

    def plot(self):
        _, ax = plt.subplots()

        for name, vals in self._vars.items():
            ax.plot(self.x, vals, label=f'{name}')

        plt.title(self.name)
        ax.legend()
        plt.show()


