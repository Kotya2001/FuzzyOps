from time import perf_counter
import unittest
import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

import numpy as np
from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.fuzzy_optimization import get_interaction_matrix, \
    LinearOptimization, check_LR_type



class TestFuzzyOptimization(unittest.TestCase):
    """
    Testing fuzzy optimization algorithms

    """

    def setUp(self) -> None:
        self.d = Domain((1, 10, 1), name='d')
        self.d1 = Domain((1, 45, 1), name='d1')
        self.d2 = Domain((1, 6, 1), name='d2')
        self.d3 = Domain((1, 15, 1), name='d3')

        self.number = self.d.create_number('triangular', 1, 2, 7, name='n1')
        self.number1 = self.d1.create_number('triangular', 2, 6, 8, name='n1')
        self.number2 = self.d2.create_number('triangular', 3, 4, 5, name='n1')
        self.number3 = self.d3.create_number('triangular', 1, 3, 6, name='n1')

        self.x = np.arange(start=0.01, stop=1, step=0.01)
        self.r = np.array([9.919, -6.175, 4.372, -3.680, 2.663, -2.227,
                           1.742, -2.789, 11.851, -8.565, 0.938, -0.103])
        self.size = self.r.shape[0]

        self.simple_C = np.array([[4, 2]])
        self.simple_b = np.array([18, 9, 10])
        self.simple_A = np.array([[2, 3], [-1, 3], [2, -1]])

        self.A = np.random.rand(1000, 10000)
        self.b = np.random.rand(1000)
        self.C = np.random.rand(500, 10000)
        self.g = np.random.rand(500)
        self.t = np.random.rand(500)

    def test_check_interactions(self):
        """
        A test to check a specific interaction for
        specific fuzzy numbers (coefficients for functions)

        """
        matrix = np.array([[self.number, self.number2],
                           [self.number1, self.number3]])
        params = np.array([[[2, 1, 7], [6, 2, 8]],
                           [[4, 3, 5], [3, 1, 6]]])

        assert check_LR_type(matrix)

        alphas, interactions_list = get_interaction_matrix(params)

        print(interactions_list)

        assert alphas[:, 0].sum() == 2

    def test_check_simple_linear_opt(self):
        """
        Linear optimization test (small task)

        """

        opt = LinearOptimization(self.simple_A, self.simple_b, self.simple_C, 'max')
        r, v = opt.solve_cpu()
        assert np.allclose(v, np.array([6, 2]))

        print(r, v)

    def test_check_complex_task_cpu(self):
        """
        Multi-criteria optimization test with a large number of criteria, variables, and CPU constraints

        """
        start = perf_counter()
        opt = LinearOptimization(self.A, self.b, self.C, 'min')
        result, _ = opt.solve_cpu()
        end = perf_counter()
        print(f"The value of the objective function: {result}")
        print('Lead time:', end - start)

    def test_check_complex_task_gpu(self):
        """
        A multi-criteria optimization test with a large number of criteria, variables, and constraints on the GPU

        """
        start = perf_counter()
        opt = LinearOptimization(self.A, self.b, self.C, 'min')
        result, vars = opt.solve_gpu()
        end = perf_counter()
        print(f"The value of the objective function: {result}")
        print('Lead time:', end - start)
