import unittest
import pandas as pd
import os
import sys
from pathlib import Path
from time import perf_counter
from sklearn.preprocessing import LabelEncoder

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_nn import Model


class TestFuzzyNN(unittest.TestCase):
    """
    Тестирование нечеткой нейронной сети (алгоритм 1 - ANFIS)
    """

    def setUp(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        classification_data = pd.read_csv(os.path.join(project_root, "tests", "Iris.csv"))
        reg_data = pd.read_csv(os.path.join(project_root, "tests", "sales.csv"))

        n_features = 2

        self.n_terms = [5, 5]
        self.n_out_vars1 = 3
        self.n_out_vars2 = 1
        self.lr = 3e-4
        self.task_type1 = "classification"
        self.task_type2 = "regression"
        self.batch_size = 2
        self.member_func_type = "gauss"
        self.epochs = 100
        self.verbose = True

        self.X_class, self.y_class = classification_data.iloc[:, 1: 1 + n_features].values, \
                                     classification_data.iloc[:, -1]

        self.X_reg, self.y_reg = reg_data.iloc[:, 1:].values, reg_data.iloc[:, 0]

    def test_classification(self):
        """
        Тестирование задачи классификации
        """
        le = LabelEncoder()
        y = le.fit_transform(self.y_class)

        model = Model(self.X_class, y,
                      self.n_terms, self.n_out_vars1,
                      self.lr,
                      self.task_type1,
                      self.batch_size,
                      self.member_func_type,
                      self.epochs,
                      self.verbose)

        # создание экземпляра класса
        model.train()
        best_score = max(model.scores)
        assert best_score > 80


class TesCpuGPU(unittest.TestCase):
    """
    Тестирование модели классификации на больших данных на GPU и на CPU
    """

    def setUp(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        classification_data = pd.read_csv(os.path.join(project_root, "tests", "Iris.csv"))

        self.n_out_vars = 3
        self.lr = 3e-4
        self.task_type = "classification"
        self.batch_size = 2
        self.member_func_type = "gauss"
        self.epochs = 20

        self.X_class, self.y_class = classification_data.iloc[:, 1:-1].values, \
                                     classification_data.iloc[:, -1]

        self.n_terms = [10] * self.X_class.shape[1]
        self.verbose = True

    def test_on_cpu(self):
        """
        Тестирование модели классификации на больших данных CPU
        """
        le = LabelEncoder()
        y = le.fit_transform(self.y_class)

        model = Model(self.X_class, y,
                      self.n_terms, self.n_out_vars,
                      self.lr,
                      self.task_type,
                      self.batch_size,
                      self.member_func_type,
                      self.epochs,
                      self.verbose)

        start = perf_counter()
        model.train()
        end = perf_counter()
        print('cpu train:', end - start)

    def test_on_gpu(self):
        """
        Тестирование модели классификации на больших данных GPU
        """
        le = LabelEncoder()
        y = le.fit_transform(self.y_class)
        model = Model(self.X_class, y,
                      self.n_terms, self.n_out_vars,
                      self.lr,
                      self.task_type,
                      self.batch_size,
                      self.member_func_type,
                      self.epochs,
                      self.verbose,
                      device="cuda")

        start = perf_counter()
        model.train()
        end = perf_counter()
        print('gpu train:', end - start)
