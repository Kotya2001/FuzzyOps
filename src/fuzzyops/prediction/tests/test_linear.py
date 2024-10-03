import unittest
from ...fuzzy_numbers import Domain
from ..linear import fit_fuzzy_linear_regression
import numpy as np

class TestFuzzyLR(unittest.TestCase):

    def testLinearRegression(self):
        fuzzyType = 'triangular'
        domain = Domain((0, 15, 0.1), method='minimax')

        X = [
            domain.create_number(fuzzyType, 1.5, 2, 2.5),
            domain.create_number(fuzzyType, 3, 3.5, 4),
            domain.create_number(fuzzyType, 4.5, 5.5, 6.5),
            domain.create_number(fuzzyType, 6.5, 7, 7.5),
            domain.create_number(fuzzyType, 8, 8.5, 9),
            domain.create_number(fuzzyType, 9.5, 10.5, 11.5),
            domain.create_number(fuzzyType, 10.5, 11, 11.5),
            domain.create_number(fuzzyType, 12, 12.5, 13),
        ]


        Y = [
            domain.create_number(fuzzyType, 3.5, 4, 4.5),
            domain.create_number(fuzzyType, 5, 5.5, 6),
            domain.create_number(fuzzyType, 6.5, 7, 8.5),
            domain.create_number(fuzzyType, 6, 6.5, 7),
            domain.create_number(fuzzyType, 8, 8.5, 9),
            domain.create_number(fuzzyType, 7, 8, 9),
            domain.create_number(fuzzyType, 10, 10.5, 11),
            domain.create_number(fuzzyType, 9, 9.5, 10),
        ]
        a, b, err = fit_fuzzy_linear_regression(X, Y)
        assert np.allclose(np.array([a, b, err]), np.array([0.66983449, 2.26870012, 1.10900925]))