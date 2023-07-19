from src.fuzzyops.fuzzy_numbers import FuzzyNumber
from .check_LR_type import check_LR_type
import numpy as np
from typing import Union

NumberTypes = Union["triangular", "gauss", "trapezoidal"]


def transform_matrix(matrix: np.ndarray[FuzzyNumber], type_of_all_number: NumberTypes):
    """
    Transform matrix for optimization and check all conditions of numbers

    :param matrix: np.ndarray[FuzzyNumber]
    :param type_of_all_number: NumberTypes
    :return: new_matrix: np.ndarray
    """
    # new_matrix = matrix.copy()
    rows, cols = matrix.shape

    for row in range(rows):
        for col in range(cols):
            number = matrix[row][col]

            if number.domain.membership_type != type_of_all_number:
                raise ValueError("Not right type of one number, create all numbers with certain type")
            if not check_LR_type(number):
                raise ValueError("Fuzzy number must be unimodal and convex")

            # bounds = np.array(number.domain.bounds)
            # if type_of_all_number == "triangular":
            #     bounds[0], bounds[1] = bounds[1], bounds[0]
            # if type_of_all_number == "trapezoidal":
            #     bounds[0], bounds[1], bounds[2] = bounds[1], bounds[2], bounds[0]
            #
            # new_matrix[row][col] = bounds

    return True


def calc_interaction_matrix(matrix: np.ndarray[FuzzyNumber]):
    k = matrix.copy()
    rows, cols = matrix.shape
    n = max(rows, cols)

    for row in range(rows):
        for col in range(cols):

            if row == col:
                k[row][col] = 1
            else:

                # numerator = np.sum(np.array([calc_scalar_value(matrix[row][l], matrix[col][l])
                #                              for l in range(n)]), axis=0)
                numerator = np.sum(np.array([matrix[row][l] * matrix[col][l]
                                             for l in range(n)]), axis=0)

                # square_sum_i = np.sum(np.array([calc_scalar_value(matrix[row][l], matrix[row][l])
                #                                 for l in range(n)]), axis=0)
                # square_sum_j = np.sum(np.array([calc_scalar_value(matrix[col][l], matrix[col][l])
                #                                 for l in range(n)]), axis=0)

                square_sum_i = np.sum(np.array([matrix[row][l] * matrix[row][l]
                                                for l in range(n)]), axis=0)
                square_sum_j = np.sum(np.array([matrix[col][l] * matrix[col][l]
                                                for l in range(n)]), axis=0)

                root1, root2 = square_sum_i.maybe, square_sum_j.maybe


                # root_i_1, root_i_2 = calc_root_value(square_sum_i)
                # root_j_1, root_j_2 = calc_root_value(square_sum_j)

                root_scalar = root1 * root2
                # print(root_scalar)
                dev_res = numerator / root_scalar
                print(dev_res)

                # # root1, root2 = calc_scalar_value(root_i_1, root_j_1), calc_scalar_value(root_i_2, root_j_2)
                # res1, res2 = dev_func(numerator, root1), dev_func(numerator, root2)
                #
                # k[row][col] = res1[0]
    return k


def calc_scalar_value(c1: np.ndarray, c2: np.ndarray):
    res = c1.copy()
    res[0], res[1], res[2] = \
        c1[0] * c2[0], c1[0] * c2[1] + c2[0] * c1[1] - c1[1] * c2[1], \
        c1[0] * c2[2] + c2[0] * c1[2] + c1[2] * c2[2]

    return res


def calc_root_value(square_num: np.ndarray):
    z1 = np.array([square_num[0] ** 0.5,
                   square_num[0] ** 0.5 + (square_num[0] - square_num[1]) ** 0.5,
                   -1 * square_num[0] ** 0.5 + (square_num[0] + square_num[2]) ** 0.5])

    z2 = np.array([square_num[0] ** 0.5 * -1,
                   ((square_num[0]) ** 0.5) * -1 + (square_num[0] - square_num[1]) ** 0.5,
                   (square_num[0]) ** 0.5 + (square_num[0] + square_num[2]) ** 0.5])
    return z1, z2


def dev_func(c1: np.ndarray, c2: np.ndarray):
    res = c1.copy()
    res[0], res[1], res[2] = c1[0] / c2[0], (c1[0] * c2[2] + c2[0] * c1[1]) / c2[0] ** 2, \
                             (c1[0] * c2[1] + c2[0] * c1[2]) / c2[0] ** 2

    return res
