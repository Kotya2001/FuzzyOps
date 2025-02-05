from fuzzyops.fuzzy_numbers import FuzzyNumber
import numpy as np
from uncertainties import ufloat
from typing import Union, Callable
from dataclasses import dataclass
import pandas as pd
import torch
from typing import Tuple

import cvxpy as cp

NumberTypes = Union["triangular"]
arrayTypes = Union[np.ndarray, torch.tensor]


class LinearOptimization:
    def __init__(self, A: np.ndarray, b: np.ndarray, C: np.ndarray, task_type):
        self.A = A
        self.b = b
        self.C = C
        self.task_type = task_type
        self.num_vars, self.num_crits, self.num_cons = A.shape[1], C.shape[0], b.shape[0]
        self.weights = np.ones(C.shape[0])

    def solve_cpu(self, all_positive=True):
        x = cp.Variable(self.num_vars)
        mus = [self.C[i] @ x for i in range(self.num_crits)]
        mus_stacked = cp.vstack(mus)
        objective_value = self.weights @ mus_stacked

        if self.task_type == "max":
            objective = cp.Maximize(objective_value)
        elif self.task_type == "min":
            objective = cp.Minimize(objective_value)
        else:
            raise ValueError

        constraints = [self.A @ x <= self.b]
        if all_positive:
            constraints.append(x >= 0)

        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        return result, x.value

    def solve_gpu(self, lr=0.001, epochs=10000):
        x = torch.randn(self.num_vars, device='cuda', requires_grad=True)
        C = torch.tensor(self.C.tolist(), dtype=torch.float32, device='cuda')
        b = torch.tensor(self.b.tolist(), dtype=torch.float32, device='cuda')
        A = torch.tensor(self.A.tolist(), dtype=torch.float32, device='cuda')

        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            mus = [torch.matmul(C[i], x) for i in range(self.num_crits)]
            if self.task_type == "max":
                objective = torch.max(torch.stack(mus))
            elif self.task_type == "min":
                objective = torch.min(torch.stack(mus))
            else:
                raise ValueError

            constraints_violation = (torch.matmul(A, x) - b).clamp(min=0).sum()
            loss = -objective + constraints_violation

            loss.backward()
            optimizer.step()

        return objective.item(), x.detach().cpu().numpy()


class RankingSolution:
    def __init__(self, A: np.ndarray, b: np.ndarray, C: np.ndarray, task_type):
        self.A = A
        self.b = b
        self.C = C
        self.task_type = task_type
        self.num_vars, self.num_crits, self.num_cons = A.shape[1], C.shape[0], b.shape[0]
        self.weights = np.ones(C.shape[0])

    def is_close(self, a, b, tolerance=1e-9):
        """ Функция для проверки, являются ли две точки "близкими" друг к другу """
        return np.allclose(a, b, atol=tolerance)

    def merge_points(self, points, tolerance=1e-9):
        """ Объединяет близкие точки в один массив """
        unique_points = []

        for point in points:
            # Проверка, есть ли уже близкая к ней точка
            if not any(self.is_close(point, existing, tolerance) for existing in unique_points):
                unique_points.append(point)

        return np.array(unique_points)

    def solve_tasks(self, device_type: str = 'cpu'):
        all_solutions = []

        for i in range(1, self.num_crits):
            c = self.C[i, :]
            new_c = c[np.newaxis, :]
            opt = LinearOptimization(self.A, self.b, new_c, self.task_type)
            if device_type == "cpu":
                r, v = opt.solve_cpu()
            elif device_type == "cuda":
                r, v = opt.solve_gpu()
            else:
                raise ValueError("Неизветсный тип девайса")

            all_solutions.append(v)

        unique_solutions = self.merge_points(all_solutions)
        return unique_solutions

    def make_table(self):
        pass


@dataclass
class Response:
    interaction_coefs: np.ndarray
    interactions: pd.DataFrame
    alphas: np.ndarray


# check types of all nums, must be the same
def _check_types(number: FuzzyNumber, type_of_all_number: NumberTypes) -> bool:
    """
    Проверяет тип нечеткого числа.

    Args:
        number (FuzzyNumber): Нечеткое число для проверки.
        type_of_all_number (NumberTypes): Ожидаемый тип для всех нечетких чисел.

    Returns:
        bool: True, если тип нечеткого числа соответствует ожидаемому типу, иначе False.
    """

    if type_of_all_number != "triangular":
        return False
    return number.domain.membership_type == type_of_all_number


# check LR type of all nums in matrix, must be convex and unimodal
def _check_LR_type(number: FuzzyNumber) -> bool:
    """
    Проверяет, соответствует ли нечеткое число LR-типу.

    Args:
        number (FuzzyNumber): Нечеткое число для проверки.

    Returns:
        bool: True, если нечеткое число является выпуклым и унимодальным, иначе False.
    """

    values = number.values.numpy()
    membership_type = number.domain.membership_type
    _mu = np.where(values == 1.0)[0]
    if membership_type == "triangular":
        return _mu.size == 1
    return False


vectorized_check_types = np.vectorize(_check_types)
vectorized_check_LR_type = np.vectorize(_check_LR_type)


# decorator for check all rules and transform matrix
def transform_matrix(func: Callable) -> Callable:
    """
    Декоратор для проверки всех условий и трансформации матрицы нечетких чисел.

    Args:
        func (Callable): Функция, которая будет вызываться после проверки условий.

    Returns:
        Callable: Обернутая функция с проверками.
    """

    def inner(matrix: np.ndarray[FuzzyNumber], type_of_all_number: NumberTypes):
        row, col = matrix.shape
        if row != col:
            raise ValueError("Matrix should be squared")
        if not np.all(vectorized_check_types(matrix, type_of_all_number)):
            raise ValueError("Not right type of one number,"
                             "all number must have the same type and must be triangular")
        if not np.all(vectorized_check_LR_type(matrix)):
            raise ValueError("Fuzzy number must be unimodal and convex")

        new_matrix = np.zeros_like(matrix)
        for index, vector in np.ndenumerate(matrix):
            bounds = vector.domain.bounds

            bounds[0], bounds[1] = bounds[1], bounds[0]

            new_matrix[index[0], index[1]] = np.array(bounds)
        # print(new_matrix)
        return func(new_matrix)

    return inner


def calc_root_value(square_num: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет корневые значения для квадратного нечеткого числа.

    Args:
        square_num (np.ndarray): Входной массив нечеткого числа, чтобы вычислить корень.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Два массива с корнями.
    """

    z1 = np.array([square_num[0] ** 0.5,
                   ufloat(square_num[0] ** 0.5, (square_num[0] - square_num[1]) ** 0.5),
                   ufloat(-1 * (square_num[0] ** 0.5), (square_num[0] + square_num[2]) ** 0.5)])

    z2 = np.array([square_num[0] ** 0.5 * -1,
                   ufloat((square_num[0] ** 0.5) * -1, (square_num[0] - square_num[1]) ** 0.5),
                   ufloat(square_num[0] ** 0.5, (square_num[0] + square_num[2]) ** 0.5)])
    return z1, z2


def calc_scalar_value(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """
    Вычисляет скалярное значение на основе двумерных массивов.

    Args:
        c1 (np.ndarray): Первый массив поэлементных коэффициентов.
        c2 (np.ndarray): Второй массив поэлементных коэффициентов.

    Returns:
        np.ndarray: Вычисленный результат как массив скалярных значений.
    """

    res = c1.copy()
    res[0], res[1], res[2] = \
        c1[0] * c2[0], c1[0] * c2[1] + c2[0] * c1[1] - c1[1] * c2[1], \
        c1[0] * c2[2] + c2[0] * c1[2] + c1[2] * c2[2]

    return res


def _define_interaction_type(table: np.ndarray,
                             k: np.ndarray,
                             total_info: dict) -> np.ndarray:
    """
    Определяет тип взаимодействия на основе значения n.

    Args:
        j (int): Индекс строки в таблице.
        table (np.ndarray): Таблица для учета количества различных типов взаимодействия.
        n (float): Значение, служащее основой для определения типа взаимодействия.

    Returns:
        np.ndarray: Обновленная таблица с подсчетами.
    """

    for index, _ in np.ndenumerate(k):
        row, col = index[0], index[1]

        if 0.5 <= k[row][col] <= 1:
            table[row][0] += 1
            total_info["Кооперация"][row].append(col)
        elif -1 <= k[row][col] <= -0.5:
            table[row][1] += 1
            total_info["Конфликт"][row].append(col)
        elif -0.5 < k[row][col] < 0.5:
            table[row][2] += 1
            total_info["Независимость"][row].append(col)

    return table, total_info


@transform_matrix
def get_interaction_matrix(matrix: np.ndarray) -> Response:
    """
    Создает коэффициенты взаимодействия между каждой функцией.
    Алгоритм реализован по статье:

      Аристова Е.М. Алгоритм решения задачи нечеткой многоцелевой линейной оптимизации
      с помощью определения коэффициента взаимодействия между
      целевыми функциями // Вестник Воронежского государственного университета.
      Серия: Системный анализ и информационные технологии. 2017 № 2. С. 105-110.


    Args:
        matrix (np.ndarray): Входная матрица нечетких чисел.

    Returns:
        Response: Объект Response, содержащий коэффициенты взаимодействия,
                  таблицу взаимодействий и альфа значения.
    """
    n = matrix.shape[0]
    k, interactions = np.zeros_like(matrix), np.zeros((n, 3))
    total_info = {"Кооперация": [[] for _ in range(n)],
                  "Конфликт": [[] for _ in range(n)],
                  "Независимость": [[] for _ in range(n)]}
    np.fill_diagonal(k, 1)
    repeats = {}
    for index, _ in np.ndenumerate(matrix):
        row, col = index[0], index[1]
        if row != col:
            total = row + col
            if str(total) not in list(repeats.keys()):

                numerator = np.sum(np.array([calc_scalar_value(matrix[row][l], matrix[col][l])
                                             for l in range(n)]), axis=0)

                square_sum_i = np.sum(np.array([calc_scalar_value(matrix[row][l], matrix[row][l])
                                                for l in range(n)]), axis=0)
                square_sum_j = np.sum(np.array([calc_scalar_value(matrix[col][l], matrix[col][l])
                                                for l in range(n)]), axis=0)

                root_i_1, root_i_2 = calc_root_value(square_sum_i)
                root_j_1, root_j_2 = calc_root_value(square_sum_j)

                root1, root2 = calc_scalar_value(root_i_1, root_j_1), calc_scalar_value(root_i_2, root_j_2)
                res = numerator[0] / root1[0]
                k[row][col] = res
                repeats.update({str(total): (row, col, res)})
            else:
                row, col, res = repeats[str(total)]
                k[col][row] = res

                del repeats[str(total)]
                continue

    interactions, interactions_list = _define_interaction_type(interactions, k, total_info)
    alphs = interactions / n

    response = Response(
        interaction_coefs=k,
        interactions=pd.DataFrame(data={"Кооперация": interactions[:, 0],
                                        "Конфликт": interactions[:, 1],
                                        "Независимость": interactions[:, 2]}),
        alphas=alphs
    )
    return response.interactions, response.interaction_coefs, response.alphas, interactions_list


def __calc(with_ind: int, indx: list[int], params: np.ndarray) -> np.ndarray:
    res = params[with_ind] * params[indx[0]]
    for i in range(1, len(indx)):
        res += (params[with_ind] + params[indx[i]])
    return res


def calc_total_functions(alphs: np.ndarray, params: np.ndarray, interaction_info: dict, n: int):
    arrays = []
    for i in range(n):
        coop_coef = alphs[i, 0]
        conflict_coef = alphs[i, 1]
        indep_coef = alphs[i, 2]

        res = coop_coef * __calc(i, interaction_info["Кооперация"][i], params) \
              + conflict_coef * __calc(i, interaction_info["Конфликт"][i], params) \
              + indep_coef * __calc(i, interaction_info["Независимость"][i], params)
        arrays.append(res)

    combined_array = np.vstack(arrays)
    return np.sum(combined_array, axis=0)
