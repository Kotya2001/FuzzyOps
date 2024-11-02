import unittest
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[3]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_clustering import fcm, fcm_predict


class TestFCM(unittest.TestCase):
    """
    Тестирование алгоритма FCM
    """

    def setUp(self) -> None:
        np.random.seed(42)
        xpts = np.zeros(0)
        ypts = np.zeros(0)
        self.cluster = np.zeros(0)
        xtest = np.zeros(0)
        ytest = np.zeros(0)

        # значения центров кластеров
        self.x_corr = [7, 1, 4]
        self.y_corr = [3, 2, 1]

        for x, y, in zip(self.x_corr, self.y_corr):
            xpts = np.concatenate((xpts, np.r_[np.random.normal(x, 0.5, 200)]))
            ypts = np.concatenate((ypts, np.r_[np.random.normal(y, 0.5, 200)]))

        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # данные для кластеризации
        self.features = np.c_[xpts, ypts].T

        for x, y, label in zip(self.x_corr, self.y_corr, [1, 2, 0]):
            xtest = np.concatenate((xtest, np.r_[np.random.normal(x, 0.05, 100)]))
            ytest = np.concatenate((ytest, np.r_[np.random.normal(y, 0.05, 100)]))
            self.cluster = np.concatenate((self.cluster, np.r_[[label] * 100]))

        self.test_data = np.c_[xtest, ytest].T

    def test_fuzzy_fcm_centers(self):
        """
        Тестирование алгоритма на соответствие центров кластеров изначально заданным
        :return: None
        """
        cntr, _, _, _, _, _, _ = fcm(
            self.features, 3, 2., error=0.005, maxiter=1000, init=None)

        expected = np.c_[self.x_corr, self.y_corr][(2, 0, 1), :]
        print("Заданные центры", expected)
        print("Рассчитанные алгоритмом", cntr)

        np.testing.assert_allclose(expected, cntr, rtol=0.1)

    def test_fuzzy_fcm_predict(self):
        """
        Тест кластеризации новых данных
        """

        print(self.features.shape)
        print(self.test_data.shape)

        train = pd.DataFrame(self.test_data)
        train.to_csv("/Users/ilabelozerov/FuzzyOps/src/fuzzyops/fuzzy_clustering/tests/test_data.csv")

        cntr, _, _, _, _, _, _ = fcm(
            self.features, 3, 2., error=0.005, maxiter=1000, init=None)

        print(cntr.shape)

        U, _, _, _, _, fpc = fcm_predict(
            self.test_data, cntr, 2., error=0.005, maxiter=1000, seed=1234)

        U2, _, _, _, _, fpc2 = fcm_predict(
            self.test_data, cntr, 2., error=0.005, maxiter=1000, seed=1234)

        print(U2.argmax(axis=0).shape)

        assert fpc == fpc2
        np.testing.assert_array_equal(U, U2)

        np.testing.assert_array_equal(self.cluster, U.argmax(axis=0))
