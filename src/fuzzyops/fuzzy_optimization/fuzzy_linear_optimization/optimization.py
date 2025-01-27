

from fuzzyops.fuzzy_numbers import FuzzyNumber
import numpy as np
from uncertainties import ufloat
from typing import Union, Callable, Tuple
from dataclasses import dataclass
import pandas as pd

NumberTypes = Union["triangular"]


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


# def _define_interaction_type(j: int, table: np.ndarray, n: float) -> np.ndarray:
#     """
#     Определяет тип взаимодействия на основе значения n.
#
#     Args:
#         j (int): Индекс строки в таблице.
#         table (np.ndarray): Таблица для учета количества различных типов взаимодействия.
#         n (float): Значение, служащее основой для определения типа взаимодействия.
#
#     Returns:
#         np.ndarray: Обновленная таблица с подсчетами.
#     """
#
#     if 0.5 <= n <= 1:
#         table[j][0] += 1
#     elif -1 <= n <= -0.5:
#         table[j][1] += 1
#     elif -0.5 < n < 0.5:
#         table[j][2] += 1
#
#     return table

def _define_interaction_type(table: np.ndarray, k: np.ndarray) -> np.ndarray:
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
        elif -1 <= k[row][col] <= -0.5:
            table[row][1] += 1
        elif -0.5 < k[row][col] < 0.5:
            table[row][2] += 1

    return table


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

    k, interactions = np.zeros_like(matrix), np.zeros((matrix.shape[0], 3))
    np.fill_diagonal(k, 1)
    n = matrix.shape[0]
    repeats = {}
    # print(matrix)
    m = np.array([[np.array([4, 2, 7]), np.array([5, 3, 4])],
                  [np.array([4, 2, 3]), np.array([3, 1, 5])]])
    matrix = m
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
                # interactions = _define_interaction_type(row, interactions, res)
                # _define_interaction_type(row, interactions, res)
                repeats.update({str(total): (row, col, res)})
            else:
                row, col, res = repeats[str(total)]
                k[col][row] = res

                # interactions = _define_interaction_type(col, interactions, res)
                # _define_interaction_type(col, interactions, res)
                del repeats[str(total)]
                continue
    interactions = _define_interaction_type(interactions, k)
    alphs = interactions / n

    response = Response(
        interaction_coefs=k,
        interactions=pd.DataFrame(data={"Кооперация": interactions[:, 0],
                                        "Конфликт": interactions[:, 1],
                                        "Независимость": interactions[:, 2]}),
        alphas=alphs
    )
    return response.interactions, response.interaction_coefs, response.alphas
