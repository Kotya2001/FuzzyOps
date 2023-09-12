"""
Проверка числе на LR-тип

Необходимо проверить нечеткое число на нормальность:

 - существует значение носителя, в котором функция принадлежности равна единице (условие нормальности);
 - при отступлении от своего максимума влево или вправо функция принадлежности не возрастает (условие выпуклости);

"""

from src.fuzzyops.fuzzy_numbers import FuzzyNumber
import numpy as np
from uncertainties import ufloat
from typing import Union

NumberTypes = Union["triangular"]


# check types of all nums, must be the same
def _check_types(number: FuzzyNumber, type_of_all_number: NumberTypes):
    if type_of_all_number != "triangular":
        return False
    return number.domain.membership_type == type_of_all_number


# check LR type of all nums in matrix, must be convex and unimodal
def _check_LR_type(number: FuzzyNumber) -> bool:
    values = number.values.numpy()
    membership_type = number.domain.membership_type
    _mu = np.where(values == 1.0)[0]
    if membership_type == "triangular":
        return _mu.size == 1
    return False


vectorized_check_types = np.vectorize(_check_types)
vectorized_check_LR_type = np.vectorize(_check_LR_type)


# decorator for check all rules and transform matrix
def transform_matrix(func):
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
        return func(new_matrix)

    return inner


def calc_root_value(square_num: np.ndarray):
    z1 = np.array([square_num[0] ** 0.5,
                   ufloat(square_num[0] ** 0.5, (square_num[0] - square_num[1]) ** 0.5),
                   ufloat(-1 * (square_num[0] ** 0.5), (square_num[0] + square_num[2]) ** 0.5)])

    z2 = np.array([square_num[0] ** 0.5 * -1,
                   ufloat((square_num[0] ** 0.5) * -1, (square_num[0] - square_num[1]) ** 0.5),
                   ufloat(square_num[0] ** 0.5, (square_num[0] + square_num[2]) ** 0.5)])
    return z1, z2


def calc_scalar_value(c1: np.ndarray, c2: np.ndarray):
    res = c1.copy()
    res[0], res[1], res[2] = \
        c1[0] * c2[0], c1[0] * c2[1] + c2[0] * c1[1] - c1[1] * c2[1], \
        c1[0] * c2[2] + c2[0] * c1[2] + c1[2] * c2[2]

    return res


def _define_interaction_type(j: int, table: np.ndarray, n: float):
    if 0.5 <= n <= 1:
        table[j][0] += 1
    elif -1 <= n <= -0.5:
        table[j][1] += 1
    elif -0.5 < n < 0.5:
        table[j][2] += 1

    return table


@transform_matrix
def get_interaction_matrix(matrix: np.ndarray):
    """
    Create interaction coef between each function
    :param matrix: np.ndarray
    :return: np.ndarray
    """
    k, interactions = np.zeros_like(matrix), np.zeros((matrix.shape[0], 3))
    np.fill_diagonal(k, 1)
    n = matrix.shape[0]
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

                interactions = _define_interaction_type(row, interactions, res)
                repeats.update({str(total): (row, col, res)})
            else:
                row, col, res = repeats[str(total)]
                k[col][row] = res

                interactions = _define_interaction_type(col, interactions, res)
                del repeats[str(total)]
                continue
    alphs = interactions / n
    return k, interactions, alphs
