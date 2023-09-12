from src.fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

from typing import Union
import numpy as np
import pandas as pd
from scipy import interpolate
from dataclasses import dataclass

tps = Union[FuzzyNumber, float]


@dataclass(frozen=True)
class FuzzyBounds:
    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float]
    x: list[str]


@dataclass
class Archive:
    k: int
    params: np.ndarray
    loss: float
    weights: float


class Optim:
    __params: int = 3

    def __init__(self, data: np.ndarray,
                 k: int, epsilon: float,
                 q: float, n_iter: int, n_ant: int, ranges: list[FuzzyBounds],
                 r: np.ndarray, R: int):

        self.n = data.shape[1] - 1
        self.p = data.shape[0]
        self.N = self.__params * R * self.n
        self.R = R
        self.X = pd.DataFrame(data={"x_" + str(i + 1): data[:, i] for i in range(self.n)})
        self.t = data[:, -1]
        self.r = r
        self.ranges = ranges
        self._low = min([f_bound.start for f_bound in self.ranges])
        self._high = max([f_bound.end for f_bound in self.ranges])

        self.k = k
        self.theta = np.reshape(np.random.uniform(low=self._low,
                                                  high=self._high,
                                                  size=(self.k * self.X.shape[0], self.N)),
                                (self.k, self.p, self.n * R,
                                 self.__params))
        self.storage = [
            Archive(k=i, params=self.theta[i, :, :, :], loss=0, weights=0) for i in range(self.k)
        ]

        self.theta.sort()
        self.eps = epsilon
        self.q = q
        self.n_iter = n_iter
        self.n_ant = n_ant

    def __f(self, theta: np.ndarray):
        mu_value = self.__construct_fuzzy_num(theta)
        prod_value = np.prod(mu_value, axis=-1)
        noise = np.random.rand(*prod_value.shape)
        prod_value += noise
        _f = np.sum(prod_value * self.r, axis=-1) / np.sum(prod_value, axis=-1)
        return _f

    def _root_mean_squared_error(self, theta):
        _f = self.__f(theta)
        return np.sqrt(
            np.sum(
                np.square(self.t - _f)
            )
        )

    def _calc_weights(self, index):
        return np.exp(-np.square(index - 1) / (2 * np.square(self.q) * np.square(self.k))) \
               / (self.q * self.k * np.sqrt(np.pi * 2))

    def continuous_ant_algorithm(self):
        for i in range(self.k):
            loss = self._root_mean_squared_error(self.storage[i].params)
            w = self._calc_weights(i)
            self.storage[i].loss = loss
            self.storage[i].weights = w

        self.storage.sort(key=lambda x: x.loss)

        return self.storage[0]

    @staticmethod
    def interp(num: FuzzyNumber, value: Union[int, float]):
        x, y = num.domain.x, num.values
        y_inter = interpolate.interp1d(x, y)
        return y_inter(value)

    def __construct_fuzzy_num(self,
                              theta: np.ndarray):
        X_f_nums = np.zeros((self.p, self.R, self.n))
        for line in self.ranges:
            x_names = line.x
            for x_name in x_names:
                dom = Domain((line.start, line.end, line.step), name=x_name)
                _, i = tuple(x_name.split("_"))
                ind = int(i) - 1
                f_nums = np.array([
                    [
                        dom.create_number("triangular", *theta[l, ind + j, :].tolist()) for j in range(0,
                                                                                                       theta.shape[1],
                                                                                                       self.n)
                    ] for l in range(theta.shape[0])
                ])
                mu_arr = np.array([
                    [
                        self.interp(f_nums[i, j], self.X[x_name][i]) for j in range(f_nums.shape[1])
                    ] for i in range(f_nums.shape[0])
                ])

                X_f_nums[:, :, ind] = mu_arr
        return X_f_nums
