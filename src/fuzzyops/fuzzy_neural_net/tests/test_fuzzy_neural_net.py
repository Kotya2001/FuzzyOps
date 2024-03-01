import unittest

from ...fuzzy_neural_net import FuzzyNNetwork
from ...fuzzy_numbers import Domain

class TestFuzzyNN(unittest.TestCase):

    def testNeuralNet(self):
        fuzzyType = 'triangular'
        nn = FuzzyNNetwork(
            [2, 2, 1],
            (-100, 100),
            'minimax',
            fuzzyType,
            'linear'
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

        X_test = [test_domain.xTest1, test_domain.xTest2]
        y_test = [0]

        nn.fit(X_train, y_train)

        result = nn.predict(X_test)

        assert result == y_test
