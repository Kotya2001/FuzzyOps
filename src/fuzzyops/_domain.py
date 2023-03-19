import torch
import matplotlib.pyplot as plt
from ._fuzzynumber import FuzzyNumber
from typing import Callable, Union, Tuple, List
from .fuzzify.mf import memberships
from inspect import signature

RealNum = Union[float, int]


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
            #assert isinstance(name, str) and name not in self._vars, 'Name must be a unique string'
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

