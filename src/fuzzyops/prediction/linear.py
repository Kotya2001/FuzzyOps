import torch
from ..fuzzy_numbers import FuzzyNumber, Domain, memberships
from typing import List, Tuple
from copy import deepcopy


# вычисление расстояния между нечеткими числвми
def fuzzy_distance(fn0, fn1):
    a0, b0 = fn0.a, fn0.b
    a1, b1 = fn1.a, fn1.b
    left_integral = ((b1 - b0 - a1 + a0) ** 2) / 3 - 2 * a0 * (((b1 - b0 - a1 + a0)) / 2 + a1) + a1 * (
            b1 - b0 - a1 + a0) + a1 ** 2 + a0 ** 2
    a0, b0 = fn0.b, fn0.c
    a1, b1 = fn1.b, fn1.c
    right_integral = ((b1 - b0 - a1 + a0) ** 2) / 3 - 2 * a0 * (((b1 - b0 - a1 + a0)) / 2 + a1) + a1 * (
            b1 - b0 - a1 + a0) + a1 ** 2 + a0 ** 2
    dist = (left_integral + right_integral) ** 0.5
    if isinstance(dist, float):
        return dist
    return dist.item()


# Вычисление интеграла произведения
def integral_of_product(a_0, b_0, a_1, b_1):
    return (1 * (((2 * b_0 - 2 * a_0) * b_1 - 2 * a_1 * b_0 + 2 * a_0 * a_1) * 1 ** 2 + (
            3 * a_0 * b_1 + 3 * a_1 * b_0 - 6 * a_0 * a_1) * 1 + 6 * a_0 * a_1)) / 6


# вычисление интеграла суммы квадратов
def integrate_sum_squares(a_0, b_0, a_1, b_1):
    return ((b_1 - a_1) ** 2) / 3 + ((b_0 - a_0) ** 2) / 3 + a_1 * (b_1 - a_1) + a_0 * (b_0 - a_0) + a_1 ** 2 + a_0 ** 2


def convert_fuzzy_number_for_lreg(n: FuzzyNumber):
    vals = n.values
    first_increasing_idx = (vals[1:] > vals[:-1]).nonzero(as_tuple=True)[0][0] + 1
    last_zero_before_increase = (vals[:first_increasing_idx] == 0).nonzero(as_tuple=True)[0][-1].item()
    max_idx = vals.argmax().item()
    first_zero_after_peak = (vals[max_idx + 1:] == 0).nonzero(as_tuple=True)[0][0] + max_idx + 1
    a, b, c = n.domain.x[last_zero_before_increase], n.domain.x[max_idx], n.domain.x[first_zero_after_peak]
    return TriFNum(n.domain, a, b, c)


class TriFNum:
    def __init__(self, domain, a, b, c):
        def left_mf(x):
            k = 1 / (b - a)
            m = -a / (b - a)
            return k * x + m

        def right_mf(x):
            k = 1 / (b - c)
            m = -c / (b - c)
            return k * x + m

        self.left_mf = left_mf
        self.right_mf = right_mf
        self.domain = domain
        self.a = a
        self.b = b
        self.c = c

    def values(self):
        left = self.left_mf(self.domain.x).clip(0, 1)
        right = self.right_mf(self.domain.x).clip(0, 1)
        values = torch.zeros_like(self.domain.x)
        p0 = torch.where(left < 1)[0]
        p1 = torch.where((left >= 1) & (right >= 1))[0]
        p2 = torch.where(right < 1)[0]
        values[p0] = left[p0]
        values[p1] = 1
        values[p2] = right[p2]
        return values

    def __add__(self, other):
        if isinstance(other, TriFNum):
            return TriFNum(self.domain, self.a + other.a, self.b + other.b, self.c + other.c)
        elif isinstance(other, int) or isinstance(other, float):
            return TriFNum(self.domain, self.a + other, self.b + other, self.c + other)
        else:
            raise NotImplementedError('can only add TriFNums or real numbers')

    def __sub__(self, other):
        if isinstance(other, TriFNum):
            return TriFNum(self.domain, self.a - other.a, self.b - other.b, self.c - other.c)
        else:
            raise NotImplementedError('can only substract TriFNums')

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            if other > 0:
                return TriFNum(self.domain, self.a * other, self.b * other, self.c * other)
            else:
                return TriFNum(self.domain, self.c * other, self.b * other, self.a * other)
        else:
            raise NotImplementedError('can only multiply by real number')

    def integrate_left(self):
        return (self.b - self.a) / 2 + self.a

    def integrate_right(self):
        return (self.b - self.c) / 2 + self.b

    def integrate(self):
        return self.integrate_right() - self.integrate_left()

    def to_fuzzy_number(self):
        return self.domain.create_number("triangular", self.a, self.b, self.c)


def fit_fuzzy_linear_regression(X: List[TriFNum], Y: List[TriFNum]) -> Tuple[float, float]:
    """
    Implemented from https://ej.hse.ru/data/2014/09/03/1316474700/%D0%A8%D0%B2%D0%B5%D0%B4%D0%BE%D0%B2.pdf
    Parameters
    ----------
    X: List of n TriFNum objects
        Input training data
    Y: List of n TriFNum objects
        Input training targets
    Returns
    -------
    a: float
    b: float
    rmse: float
    """
    n = len(X)
    if isinstance(X[0], FuzzyNumber):
        # X = deepcopy(X)
        for i in range(n):
            X[i] = convert_fuzzy_number_for_lreg(X[i])
    if isinstance(Y[0], FuzzyNumber):
        # Y = deepcopy(Y)
        for i in range(n):
            Y[i] = convert_fuzzy_number_for_lreg(Y[i])

    I1 = sum([integral_of_product(Y[i].a, Y[i].b, X[i].a, X[i].b) for i in range(n)])
    I2 = sum([integral_of_product(Y[i].b, Y[i].c, X[i].b, X[i].c) for i in range(n)])
    J1 = sum([integral_of_product(Y[i].a, Y[i].b, X[i].b, X[i].c) for i in range(n)])
    J2 = sum([integral_of_product(Y[i].b, Y[i].c, X[i].a, X[i].b) for i in range(n)])
    K1 = sum([integral_of_product(X[i].a, X[i].b, X[i].a, X[i].b) for i in range(n)])
    K2 = sum([integral_of_product(X[i].b, X[i].c, X[i].b, X[i].c) for i in range(n)])
    L1 = sum([X[i].integrate_left() for i in range(n)])
    L2 = sum([X[i].integrate_right() for i in range(n)])
    M1 = sum([Y[i].integrate_left() for i in range(n)])
    M2 = sum([Y[i].integrate_right() for i in range(n)])
    N = sum([integrate_sum_squares(Y[i].a, Y[i].b, Y[i].b, Y[i].c) for i in range(n)])
    a_pos = max(0, (2 * n * (I1 + I2) - (L1 + L2) * (M1 + M2)) / (2 * n * (K1 + K2) - (L1 + L2) ** 2))
    a_neg = min(0, (2 * n * (J1 + J2) - (L1 + L2) * (M1 + M2)) / (2 * n * (K1 + K2) - (L1 + L2) ** 2))
    b_pos = 1 / 2 / n * (M1 + M2) - 1 / 2 / n * a_pos * (L1 + L2)
    b_neg = 1 / 2 / n * (M1 + M2) - 1 / 2 / n * a_neg * (L1 + L2)

    # Вычисление функционала H в зависимости от коэффициента а
    H_pos = a_pos ** 2 * (K1 + K2) + 2 * a_pos * b_pos * (L1 + L2) + 2 * n * b_pos ** 2 - 2 * a_pos * (
            I1 + I2) - 2 * b_pos * (M1 + M2) + N
    H_neg = a_neg ** 2 * (K1 + K2) + 2 * a_neg * b_neg * (L1 + L2) + 2 * n * b_neg ** 2 - 2 * a_neg * (
            J1 + J2) - 2 * b_neg * (M1 + M2) + N

    if H_pos < H_neg:
        a, b = a_pos, b_pos
    else:
        a, b = a_neg, b_neg
    a, b = a.item(), b.item()
    errors = []
    for i in range(len(X)):
        errors.append(fuzzy_distance(X[i] * a + b, Y[i]))
    return a, b, (sum(errors) / n) ** 0.5


if __name__ == "__main__":
    domain = Domain((0, 10, 0.1), method='minimax')

    X = [
        TriFNum(domain, 1.5, 2, 2.5),
        TriFNum(domain, 3, 3.5, 4),
        TriFNum(domain, 4.5, 5.5, 6.5),
        TriFNum(domain, 6.5, 7, 7.5),
        TriFNum(domain, 8, 8.5, 9),
        TriFNum(domain, 9.5, 10.5, 11.5),
        TriFNum(domain, 10.5, 11, 11.5),
        TriFNum(domain, 12, 12.5, 13),
    ]

    Y = [
        TriFNum(domain, 3.5, 4, 4.5),
        TriFNum(domain, 5, 5.5, 6),
        TriFNum(domain, 6.5, 7, 8.5),
        TriFNum(domain, 6, 6.5, 7),
        TriFNum(domain, 8, 8.5, 9),
        TriFNum(domain, 7, 8, 9),
        TriFNum(domain, 10, 10.5, 11),
        TriFNum(domain, 9, 9.5, 10),
    ]

    a, b = fit_fuzzy_linear_regression(X, Y)
    print('predicted values:', a, b)
    errors = []
    for i in range(len(X)):
        errors.append(fuzzy_distance(X[i] * a + b, Y[i]))
    print('mean distance:', sum(errors) / len(errors))
