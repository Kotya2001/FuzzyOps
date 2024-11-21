import unittest
import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.prediction import fit_fuzzy_linear_regression, fuzzy_distance, convert_fuzzy_number_for_lreg
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
        print(a, b, err)
        assert np.allclose(np.array([a, b, err]), np.array([0.66983449, 2.26870012, 1.10900925]))

    def testPrediction(self):
        fuzzyType = 'triangular'
        domain = Domain((0, 15, 0.1), method='minimax')
        X_test = convert_fuzzy_number_for_lreg(domain.create_number(fuzzyType, 12, 12.5, 13))
        Y_test = convert_fuzzy_number_for_lreg(domain.create_number(fuzzyType, 9, 9.5, 10))

        X = [
            domain.create_number(fuzzyType, 1.5, 2, 2.5),
            domain.create_number(fuzzyType, 3, 3.5, 4),
            domain.create_number(fuzzyType, 4.5, 5.5, 6.5),
            domain.create_number(fuzzyType, 6.5, 7, 7.5),
            domain.create_number(fuzzyType, 8, 8.5, 9),
            domain.create_number(fuzzyType, 9.5, 10.5, 11.5),
            domain.create_number(fuzzyType, 10.5, 11, 11.5),
        ]

        Y = [
            domain.create_number(fuzzyType, 3.5, 4, 4.5),
            domain.create_number(fuzzyType, 5, 5.5, 6),
            domain.create_number(fuzzyType, 6.5, 7, 8.5),
            domain.create_number(fuzzyType, 6, 6.5, 7),
            domain.create_number(fuzzyType, 8, 8.5, 9),
            domain.create_number(fuzzyType, 7, 8, 9),
            domain.create_number(fuzzyType, 10, 10.5, 11),
        ]

        a, b, err = fit_fuzzy_linear_regression(X, Y)
        Y_pred = (X_test * a) + b
        one_err = fuzzy_distance(Y_pred, Y_test)
        print(Y_pred.to_fuzzy_number(), one_err)
        assert one_err <= 2.5