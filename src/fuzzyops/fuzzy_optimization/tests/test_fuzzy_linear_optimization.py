import unittest

import numpy as np
from src.fuzzyops.fuzzy_numbers import Domain
from src.fuzzyops.fuzzy_optimization import Optim, FuzzyBounds


def __gaussian_f(mu: np.ndarray, sigma: np.ndarray):
    return np.random.default_rng().normal(loc=mu, scale=sigma)


vector_gaussian_f = np.vectorize(__gaussian_f)


class TestFuzzyLinearOptimization(unittest.TestCase):

    def setUp(self) -> None:
        self.d = Domain((1, 10, 1), name='d')
        self.d1 = Domain((1, 45, 1), name='d1')
        self.d2 = Domain((1, 6, 1), name='d2')
        self.d3 = Domain((1, 15, 1), name='d3')

        self.number = self.d.create_number('triangular', 1, 2, 7, name='n1')
        self.number1 = self.d1.create_number('triangular', 2, 6, 8, name='n1')
        self.number2 = self.d2.create_number('triangular', 3, 4, 5, name='n1')
        self.number3 = self.d3.create_number('triangular', 1, 3, 6, name='n1')

    def test_check_LR_type(self):
        # matrix = np.array([[self.number, self.number2],
        #                    [self.number1, self.number3]])
        #
        # matrix1 = torch.Tensor([[self.number, self.number2],
        #                         [self.number1, self.number3]])
        x = np.array([
            [3, 5, 6, 6, 5, 7],
            [3, 2, 0.34, 5, 4, 9]
        ])

        opt = Optim(
            data=x,
            k=3,
            q=3,
            epsilon=0.01,
            n_iter=100,
            ranges=[FuzzyBounds(start=0, step=2, end=12, x=["x_1", "x_2", "x_3", "x_4", "x_5"])],
            r=np.array([8, 10, 9]),
            R=3,
            n_ant=75
        )

        res = opt.continuous_ant_algorithm()
        print(res)
        # _w = np.array([archive.weights for archive in res])
        # theta = np.array([archive.params for archive in res])
        # sigma = np.zeros((2, 15, 3))
        # for j in range(3 - 1):
        #     sub = np.abs(theta[0, :, :, :] - theta[j + 1, :, :, :])
        #     sigma += sub
        #
        # sigma *= (0.001 / (3 - 1))
        #
        # print(_w)
        # print(theta[0, 0, 0, 0])
        #
        # for j in range(3):
        #     theta[j, :, :, :] = vector_gaussian_f(theta[j, :, :, :], sigma) * _w[j]
        #
        # print(theta[0, 0, 0, 0])
        # print(theta)

        return
