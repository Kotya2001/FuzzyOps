import unittest

import numpy as np
import matplotlib.pyplot as plt
from src.fuzzyops.fuzzy_numbers import Domain
from src.fuzzyops.fuzzy_optimization import transform_matrix, calc_interaction_matrix


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

        # self.number2 = self.d.create_number('trapezoidal', -1, -0.5, 0, 1, name='n2')
        # self.number3 = self.d.create_number('gauss', 1, 0, name='n3')

    def test_check_LR_type(self):
        # print(check_LR_type(self.number))
        # print(self.number2.domain.bounds)
        matrix = np.array([[self.number, self.number2],
                           [self.number1, self.number3]])
        # _, ax = plt.subplots()
        # print(self.number.values)
        # print(self.number.domain.x)
        # ax.plot(self.number.domain.x.numpy(), self.number.values.numpy())
        # plt.show()

        flag = transform_matrix(matrix, type_of_all_number="triangular")
        print(flag)
        if flag:
            print(calc_interaction_matrix(matrix))

        return
