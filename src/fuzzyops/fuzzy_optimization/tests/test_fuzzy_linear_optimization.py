import unittest

import numpy as np
from src.fuzzyops.fuzzy_numbers import Domain
from src.fuzzyops.fuzzy_optimization import Optim, FuzzyBounds



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
            k=2,
            q=3,
            epsilon=0.01,
            n_iter=100,
            ranges=[FuzzyBounds(start=0, step=2, end=12, x=["x_1", "x_2", "x_3", "x_4", "x_5"])],
            r=np.array([8, 10, 9]),
            R=3,
            n_ant=25
        )

        print(opt.continuous_ant_algorithm())



        return
