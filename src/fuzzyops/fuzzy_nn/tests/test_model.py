import unittest
import pandas as pd

from src.fuzzyops.fuzzy_nn import Model


class TestFuzzyNN(unittest.TestCase):
    """
    Тестирование нечеткой нейронной сети
    """
    def setUp(self) -> None:
        classification_data = pd.read_csv("data/Iris.csv")
        reg_data = pd.read_csv("data/sales.csv")
        n_features = 2

        self.n_terms = [5, 5]
        self.n_out_vars1 = 3
        self.n_out_vars2 = 1
        self.lr = 3e-4
        self.task_type1 = "classification"
        self.task_type2 = "regression"
        self.batch_size = 2
        self.member_func_type = "gauss"
        self.epochs = 500

        self.X_class, self.y_class = classification_data.iloc[:, 1: 1 + n_features].values, \
                                     classification_data.iloc[:, -1]

        self.X_reg, self.y_reg = reg_data.iloc[:, 1:].values, reg_data.iloc[:, 0]

    def test_classification(self):
        """
        Тестирование задачи классификации
        :return: None
        """
        model = Model(self.X_class, self.y_class,
                      self.n_terms, self.n_out_vars1,
                      self.lr,
                      self.task_type1,
                      self.batch_size,
                      self.member_func_type,
                      self.epochs)
        model.train()
        best_score = max(model.scores)

        assert best_score > 80

    def test_regression(self):
        """
        Тестирование задачи регрессии
        :return: None
        """
        model = Model(self.X_reg, self.y_reg,
                      self.n_terms, self.n_out_vars2,
                      self.lr,
                      self.task_type2,
                      self.batch_size,
                      self.member_func_type,
                      self.epochs)

        model.train()
        best_score = min(model.scores)

        assert best_score > 1
