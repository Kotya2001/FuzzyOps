import torch
from ..fuzzy_numbers import FuzzyNumber, Domain, memberships
from typing import List

def fuzzy_error(y: list[FuzzyNumber], y_hat: List[FuzzyNumber]):
    if len(y) != len(y_hat):
        raise ValueError("Lists must have the same size")
    errors = []
    for i in range(len(y)):
        errors.append((y - y_hat).defuzz() ** 2)
    error = (sum(errors) / len(errors)) ** 0.5
    return error

class LinearRegression:
    def __init__(self, *, copy_X=True, fuzzy_b=False):
        self.copy_X = copy_X
        self.fuzzy_b = fuzzy_b

    def fit(self, X, y):
        """
        Fit fuzzy linear model.

        Parameters
        ----------
        X: {array-like} containing FuzzyNumbers of shape {n_samples, 1}
            Training data

        y: {array-like} containing FuzzyNumbers of shape {n_samples,}
            Target values

        Returns
        -------
        self: LinearRegression
        """
        if self.fuzzy_b:
            raise NotImplementedError("Linear Regression with fuzzy coefficient is not implemented yet")
        
        # TODO copy X

        # TODO validate data

        # TODO optional: scale data

        # prediction
        n_samples = len(y)
        us = [
            x_val - sum(X) / n_samples for x_val in X
        ]
        vs = [
            y_val - sum(y) / n_samples for y_val in y
        ]
        Is = [
            sum(us[i] * vs[i]) / n_samples for i in range(n_samples)
        ]
        Ks = [
            sum(us[i] * us[i]) / n_samples for i in range(n_samples)
        ]
        Ls = [

        ]
        Ms = [

        ]
        a_pos = max(0, (2 * n_samples * sum(Is) - sum(Ls) * sum(Ms))/ (2 * n_samples * sum(Ks) - sum(Ls)))

        bs_pos = 1 / 2 / n_samples * sum(Ms) - 1 / 2 / n_samples * a_pos (sum(Ls))
        H_pos = sum(([y[i] - a_pos * X[i, 0] - bs_pos[i] for i in range(n_samples)]) ** 2)


if __name__ == "__main__":
    d = Domain((0, 100))
    x1 = d.create_number('triangular', 10, 15, 20, name='x1')
    x2 = d.create_number('triangular', 11, 13, 20, name='x2')
    y1 = d.create_number('triangular', 1, 1, 2, name='y1')
    y2 = d.create_number('triangular', 1, 2, 2, name='y2')
    reg = LinearRegression()
