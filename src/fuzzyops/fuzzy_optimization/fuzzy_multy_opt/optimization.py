from typing import Tuple

import cvxpy as cp
import numpy as np

from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.elementwise.minimum import minimum


# Функция принадлежности нечеткому множеству
def _mu(f: MulExpression, g_val: np.int64, t_val: np.int64) -> minimum:
    """
    Вычисляет функцию принадлежности нечеткому множеству для заданных параметров.

    Args:
        f (MulExpression): Линейное выражение, представляющее результат умножения переменных.
        g_val (np.int64): Значение, к которому производится сравнение.
        t_val (np.int64): Значение, определяющее степень принадлежности.

    Returns:
        minimum: Значение функции принадлежности нечеткому множеству, ограниченное единицей.
    """

    return cp.minimum(1, 1 - cp.abs(f - g_val) / t_val)


def solve_problem(A: np.ndarray, b: np.ndarray,
                  C: np.ndarray, g: np.ndarray,
                  t: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Формулирует и решает задачу нечеткой оптимизации с заданными матрицами состояния и ограничениями.

    Args:
        A (np.ndarray): Матрица коэффициентов для ограничений.
        b (np.ndarray): Вектор правых частей для ограничений.
        C (np.ndarray): Матрица коэффициентов для критических значений.
        g (np.ndarray): Вектор критических значений.
        t (np.ndarray): Вектор значений, определяющих степень допуска.

    Returns:
        Tuple[float, np.ndarray]: Кортеж, содержащий значение целевой функции (максимизированное значение) и
                                   оптимальные значения переменных (вектор x).
    """

    num_vars, num_crits, num_cons = A.shape[1], C.shape[0], b.shape[0]

    # Создание переменной для оптимизации
    x = cp.Variable(num_vars)
    # Вспомогательные переменные для моделирования абсолютной величины
    # delta = cp.Variable((num_crits, 1))

    # mus = [_mu(C[i] @ x, g[i], t[i]) for i in range(num_crits)]
    mus = [C[i] @ x for i in range(num_crits)]  # Используем @, если C[i] является 2D
    mus_stacked = cp.vstack(mus)
    objective = cp.Maximize(mus_stacked)
    # objective = cp.Maximize(cp.min(mus_stacked))

    # Добавление ограничений
    constraints = [
        A @ x <= b,
        x >= 0
        # C @ x >= g - t @ delta,
        # C @ x <= g + t @ delta,
        # delta >= 0
    ]

    # Формулировка и решение задачи
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return result, x.value
