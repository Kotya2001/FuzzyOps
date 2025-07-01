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
    Fuzzy neural Network Testing (ANFIS algorithm)

    """

    def setUp(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        classification_data = pd.read_csv(os.path.join(project_root, "tests", "Iris.csv"))

        n_features = 2

        self.n_terms = [5, 5]
        self.n_out_vars1 = 3
        self.n_out_vars2 = 1
        self.task_type = "classification"
        self.lr = 3e-4
        self.batch_size = 2
        self.member_func_type = "gauss"
        self.epochs = 100
        self.verbose = True

        self.X_class, self.y_class = classification_data.iloc[:, 1: 1 + n_features].values, \
                                     classification_data.iloc[:, -1]


    def test_classification(self):
        """
        Testing the classification problem

        """
        le = LabelEncoder()
        y = le.fit_transform(self.y_class)

        model = Model(self.X_class, y,
                      self.n_terms, self.n_out_vars1,
                      self.lr,
                      self.task_type,
                      self.batch_size,
                      self.member_func_type,
                      self.epochs,
                      self.verbose)

        model.train()
        best_score = max(model.scores)
        assert best_score > 80


class TesCpuGPU(unittest.TestCase):
    """
    Testing a classification model on big data on GPU and CPU

    """

    def setUp(self) -> None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        classification_data = pd.read_csv(os.path.join(project_root, "tests", "Iris.csv"))

        self.n_out_vars = 3
        self.lr = 3e-4
        self.batch_size = 2
        self.member_func_type = "gauss"
        self.task_type = "classification"
        self.epochs = 20

        self.X_class, self.y_class = classification_data.iloc[:, 1:-1].values, \
                                     classification_data.iloc[:, -1]

        self.n_terms = [10] * self.X_class.shape[1]
        self.verbose = True

    def test_on_cpu(self):
        """
        Testing a classification model CPU

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
        Testing a classification model on GPU

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
