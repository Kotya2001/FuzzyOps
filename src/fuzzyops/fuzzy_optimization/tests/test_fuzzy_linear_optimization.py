import unittest

import numpy as np
from src.fuzzyops.fuzzy_numbers import Domain
from src.fuzzyops.fuzzy_optimization import AntOptimization, FuzzyBounds, get_interaction_matrix
import json


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

    def test_approximation(self):
        """
        Тест на проверку ошибки нечетйо оптимизации при аппроксимации функции
        :return: None
        """

        X = np.random.choice(self.x, size=self.size)
        X = np.reshape(X, (self.size, 1))
        data = np.hstack((X, np.reshape(self.r, (self.size, 1))))

        opt = AntOptimization(
            data=data,
            k=5,
            q=0.05,
            epsilon=0.005,
            n_iter=100,
            ranges=[FuzzyBounds(start=0.01, step=0.01, end=1, x=["x_1"])],
            r=self.r,
            R=12,
            n_ant=55
        )
        _ = opt.continuous_ant_algorithm()
        loss = opt.best_result.loss

        assert loss <= 1.5

    def test_check_interactions(self):
        """
        Тест на проверку конкретного взаимодействия для
        конкретных нечектиких чисел (коэффициентах при функиях)
        :return: None
        """
        matrix = np.array([[self.number, self.number2],
                           [self.number1, self.number3]])

        interactions, interaction_coefs, alphas = get_interaction_matrix(matrix, type_of_all_number="triangular")

        print(interactions)
        print(interaction_coefs)

        res = {
            "interactions": interactions.to_dict(),
            "interaction_coefs": interaction_coefs.tolist(),
            "alphas": alphas.tolist()
        }

        with open("res.json", "w") as file:
            file.write(json.dumps(res, indent=4, ensure_ascii=False))

        assert interactions["Кооперация"].sum() == matrix.shape[1]
