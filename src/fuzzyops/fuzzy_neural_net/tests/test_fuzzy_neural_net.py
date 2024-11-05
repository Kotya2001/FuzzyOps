import unittest

from time import perf_counter
from random import uniform

import os
import sys
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[3]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_neural_net import FuzzyNNetwork
from fuzzyops.fuzzy_numbers import Domain

import pandas as pd


class TestFuzzyNN(unittest.TestCase):

    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        self.data_path = os.path.join(self.project_root, "fuzzy_neural_net", "tests", "ex.csv")
        self.bigDatasetBegin = 20
        self.bigDatasetEnd = 3
        self.bigDatasetShape = [self.bigDatasetBegin, 20, 10, self.bigDatasetEnd]
        self.bigDatasetSize = 1000
        self.bigDatasetSteps = 100

    def testNeuralNetPrediction(self):
        # проверка правильности вычислений
        fuzzyType = 'triangular'
        nn = FuzzyNNetwork(
            [2, 2, 1],
            (-100, 100),
            'minimax',
            fuzzyType,
            'linear',
            False,
            False,
        )

        test_domain = Domain((-100, 100), name='test_domain', method='minimax')

        # создание набора данных

        test_domain.create_number(fuzzyType, -1, 0, 1, name='x11')
        test_domain.create_number(fuzzyType, 0, 1, 2, name='x12')
        test_domain.create_number(fuzzyType, -1, 0, 1, name='y1')

        test_domain.create_number(fuzzyType, -1, 0, 1, name='x21')
        test_domain.create_number(fuzzyType, -1, 0, 1, name='x22')
        test_domain.create_number(fuzzyType, 0, 1, 2, name='y2')

        test_domain.create_number(fuzzyType, 0, 1, 2, name='x31')
        test_domain.create_number(fuzzyType, -1, 0, 1, name='x32')
        test_domain.create_number(fuzzyType, -1, 0, 1, name='y3')

        test_domain.create_number(fuzzyType, 0, 1, 2, name='xTest1')
        test_domain.create_number(fuzzyType, 0, 1, 2, name='xTest2')
        test_domain.create_number(fuzzyType, 0, 1, 2, name='yTest')

        X_train = [
            [test_domain.x11, test_domain.x12],
            [test_domain.x21, test_domain.x22],
            [test_domain.x31, test_domain.x32],
        ]

        y_train = [
            [test_domain.y1],
            [test_domain.y2],
            [test_domain.y3],
        ]

        x = pd.DataFrame(data={"name": ["x11", "x12", "y1",
                                        "x21", "x22", "y2",
                                        "x31", "x32", "y3"],
                               "start": [-1, 0, -1,
                                         -1, 0, 1,
                                         1, 1, -1], "step": [1, 0, 1,
                                                             0, 1, 1,
                                                             1, 0, 0], "end": [2, 1, 2,
                                                                               1, 2, 2,
                                                                               1, 1, 2]})
        x.to_csv(self.data_path)

        X_test = [test_domain.xTest1, test_domain.xTest2]
        y_test = [0]

        nn.fit(X_train, y_train)

        result = nn.predict(X_test)

        print(result)

        assert result == y_test

    def testNeuralNetCPUSpeed(self):
        # проверка скорости на CPU
        fuzzyType = 'triangular'
        nn = FuzzyNNetwork(
            self.bigDatasetShape,
            (-100, 100),
            'minimax',
            fuzzyType,
            'linear',
            False,
            True,
        )

        test_domain = Domain((-100, 100), name='test_domain', method='minimax')
        test_domain.to('cpu')

        # создание набора данных
        X_train = []
        y_train = []
        for test_value in range(self.bigDatasetSize):
            x_value = []
            y_value = []
            for x_num in range(self.bigDatasetBegin):
                name = f"x_{test_value}_{x_num}"
                test_domain.create_number(fuzzyType, uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                x_value.append(getattr(test_domain, name))
            X_train.append(x_value)
            for y_num in range(self.bigDatasetEnd):
                name = f"x_{test_value}_{y_num}"
                test_domain.create_number(fuzzyType, uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                y_value.append(getattr(test_domain, name))
            y_train.append(y_value)

        print("Начало расчетов на CPU")
        start = perf_counter()

        nn.fit(X_train, y_train, self.bigDatasetSteps)

        end = perf_counter()

        print('Скорость выполнения на CPU:', end - start)

    def testNeuralNetGPUSpeed(self):
        # проверка скорости на GPU
        fuzzyType = 'triangular'
        nn = FuzzyNNetwork(
            self.bigDatasetShape,
            (-100, 100),
            'minimax',
            fuzzyType,
            'linear',
            True,
            True,
        )

        test_domain = Domain((-100, 100), name='test_domain', method='minimax')
        test_domain.to('cuda')

        # создание набора данных
        X_train = []
        y_train = []
        for test_value in range(self.bigDatasetSize):
            x_value = []
            y_value = []
            for x_num in range(self.bigDatasetBegin):
                name = f"x_{test_value}_{x_num}"
                test_domain.create_number(fuzzyType, uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                x_value.append(getattr(test_domain, name))
            X_train.append(x_value)
            for y_num in range(self.bigDatasetEnd):
                name = f"x_{test_value}_{y_num}"
                test_domain.create_number(fuzzyType, uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                y_value.append(getattr(test_domain, name))
            y_train.append(y_value)

        print("Начало расчетов на GPU")
        start = perf_counter()

        nn.fit(X_train, y_train, self.bigDatasetSteps)

        end = perf_counter()

        print('Скорость выполнения на GPU:', end - start)
