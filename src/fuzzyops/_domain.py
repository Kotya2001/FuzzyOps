import numpy as np
import matplotlib.pyplot as plt
from ._fuzzynumber import FuzzyNumber
from typing import Callable, Union
from .fuzzify.mf import memberships
from numba import cuda


class Domain:
    """Domain that represents a set of possible values of a number

    """
    def __init__(self, start, end, step=0.1, name=None, method='minimax'):
        self._x = np.arange(start, end, step)
        self.step = step
        self._dx = None
        self.name = name
        self.ling_vars = {}
        self._method = method
        self.vars = []
        self._on_cuda = False

    @property
    def cuda(self):
        return self._on_cuda

    def to_device(self):
        if not self.cuda:
            self._dx = cuda.to_device(self._x)
            self._on_cuda = True

    def to_host(self):
        if self.cuda:
            self._x = self._dx.copy_to_host()
            self._on_cuda = False

    @property
    def x(self):
        return self._x

    def add_linguistic(self, name: str, mf: np.ndarray):
        self.ling_vars[name] = mf

    def create_number(self, membership: Union[str, Callable], *args: Union[int, float], to_vars=True):
        if isinstance(membership, str):
            membership = memberships[membership]
        y = membership(self._x, *args)
        f = FuzzyNumber(self, y, self._method)
        if to_vars:
            self.vars.append(f)
        return f

    def get_linguistic(self, name: str):
        if name in self.ling_vars:
            return FuzzyNumber(self, self.ling_vars[name], method=self._method)
        else:
            raise KeyError

    def plot(self, plot_vars: bool = True):
        _, ax = plt.subplots()
        for name, y in self.ling_vars.items():
            ax.plot(self.x, y, label=name)

        if plot_vars:
            for idx, f in enumerate(self.vars):
                ax.plot(self.x, f.values, label=f'var{idx}')

        plt.title(self.name)
        ax.legend()
        plt.show()

