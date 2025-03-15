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
from fuzzyops.fuzzy_optimization import AntOptimization, FuzzyBounds, get_interaction_matrix, \
    solve_problem, LinearOptimization, check_LR_type


def f(x: np.ndarray):
    return (1 + 10 * np.exp(100 * np.square(x - 0.7))) * ((np.sin(125 / (x + 1.5))) / x + 0.1)


class TestFuzzyOptimization(unittest.TestCase):
    """
    Тестирование алгоритмов нечеткиой оптимизации
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

    def test_approximation(self):
        """
        Тест на проверку ошибки нечеткой оптимизации при аппроксимации функции
        """

        X = np.random.choice(self.x, size=self.size)
        X = np.reshape(X, (self.size, 1))
        data = np.hstack((X, np.reshape(self.r, (self.size, 1))))

        array = np.arange(12)
        rules = array.reshape(12, 1)

        opt = AntOptimization(
            data=data,
            k=5,
            q=0.8,
            epsilon=0.005,
            n_iter=100,
            ranges=[FuzzyBounds(start=0.01, step=0.01, end=1, x="x_1")],
            r=self.r,
            n_terms=12,
            n_ant=55,
            mf_type="triangular",
            base_rules_ind=rules
        )
        _ = opt.continuous_ant_algorithm()
        print(opt.best_result)
        loss = opt.best_result.loss

        assert loss <= 1.7

    def test_check_interactions(self):
        """
        Тест на проверку конкретного взаимодействия для
        конкретных нечектиких чисел (коэффициентах при функиях)
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
        Тест линейной оптимизации
        """

        opt = LinearOptimization(self.simple_A, self.simple_b, self.simple_C, 'max')
        r, v = opt.solve_cpu()
        assert np.allclose(v, np.array([6, 2]))

        print(r, v)

    def test_check_complex_task_cpu(self):
        """
        Тест многокритериальной оптимизации с нечеткой целью
        при большом числе критериев, переменных и ограничений
        """
        start = perf_counter()
        opt = LinearOptimization(self.A, self.b, self.C, 'min')
        result, vars = opt.solve_cpu()
        end = perf_counter()
        print(f"Значение целевой функции: {result}")
        print('Время выполнения:', end - start)

    def test_check_complex_task_gpu(self):
        """
        Тест многокритериальной оптимизации с нечеткой целью
        при большом числе критериев, переменных и ограничений
        """
        start = perf_counter()
        opt = LinearOptimization(self.A, self.b, self.C, 'min')
        result, vars = opt.solve_gpu()
        end = perf_counter()
        print(f"Значение целевой функции: {result}")
        print('Время выполнения:', end - start)
