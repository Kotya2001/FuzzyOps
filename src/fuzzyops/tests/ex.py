"""
Задача:
    Необходимо смоделировать вероятность проникновения злоумышленников в корпоративную систему, а также
    вероятность рапспрстарнения атаки при помощи построения базы правил и нечеткого логического вывода

Нечеткие переменные:
    X1 - Число активных пользователей в системе
    X2 - Время (часы)
    Y1 - Восможность проникновения
    Y2 - Возможность распространения атаки

База правил:
    Если пользователей (Х1) "несколько" И время (Х2) "нерабочее", То Возможность проникновения (Y1) "средняя";
    Если пользователей (Х1) "много" И время (Х2) "рабочее", То Возможность проникновения (Y1) "низкая";
    Если пользователей (Х1) "много" И время (Х2) "нерабочее", То Возможность проникновения (Y1) "высокая";

    Если пользователей (Х1) "несколько" И время (Х2) "нерабочее", То Возможность распространения атаки (Y2) "средняя";
    Если пользователей (Х1) "много" И время (Х2) "рабочее", То Возможность распространения атаки (Y2) "низкая";
    Если пользователей (Х1) "много" И время (Х2) "несколько", То Возможность распространения атаки (Y2) "высокая";

"""

import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_optimization import solve_problem, RankingSolution, LinearOptimization, calc_total_functions
from fuzzyops.fuzzy_numbers import Domain

import numpy as np
import pandas as pd

coefs_domain = Domain((-20, 50, 1), name='coefs')
n = 9

f_num_params = np.array([
    [[1, 4, 7], [0, 2, 5]],
    [[1, 2, 3], [2, 4, 6]],
    [[-1, 3, 4], [5, 9, 12]],
    [[7, 8, 10], [1, 2, 6]],
    [[3, 4, 8], [-2, -1, 1]],
    [[1, 3, 6], [-4, -2, 2]],
    [[-3, -2, 0], [3, 4, 4]],
    [[-4, -3, -1], [-1, 1, 2]],
    [[-7, -4, -2], [-6, -3, -1]]
])


# берем модальные значения
C = f_num_params[:, :, 1]


for i in range(n):
    print(f_num_params[i])

for i in range(n):
    arr = np.array()
    coefs_domain.create_number('triangular', 1, 4, 7, name=f'c_{i}_1')

A = np.array([[2, 3],
              [-1, 3],
              [2, -1]])

b = np.array([18, 9, 10])

k = np.array([[1, 0.8, 0.71, 0.98, 0.25, 0.49, 0, -0.71, -0.99],
              [0.8, 1, 0.99, 0.65, 0.22, -0.12, 0.6, -0.14, -0.89],
              [0.71, 0.99, 1, 0.54, 0.08, -0.26, 0.71, 0, -0.82],
              [0.98, 0.65, 0.54, 1, 0.88, 0.67, -0.22, -0.84, -0.97],
              [0.25, 0.22, 0.08, 0.88, 1, 0.94, -0.65, -0.99, -0.63],
              [0.49, -0.12, -0.26, 0.67, 0.94, 1, -0.87, -0.96, -0.33],
              [0, 0.6, 0.71, -0.22, -0.65, -0.87, 1, 0.71, -0.18],
              [-0.71, -0.14, 0, -0.84, -0.99, -0.96, 0.71, 1, 0.57],
              [-0.99, -0.89, -0.82, -0.97, -0.63, -0.33, -0.18, 0.57, 1]])

C = np.array([[4, 2],
              [2, 4],
              [3, 9],
              [8, 2],
              [4, -1],
              [3, -2],
              [-2, 4],
              [-3, 1],
              [-4, -3]])

n = 9
interactions = np.zeros((n, 3))
total_info = {"Кооперация": [[] for _ in range(n)],
              "Конфликт": [[] for _ in range(n)],
              "Независимость": [[] for _ in range(n)]}


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


interactions, interactions_list = _define_interaction_type(interactions, k, total_info)


def calc(with_ind, indx, params):
    res = params[with_ind] * params[indx[0]]
    for i in range(1, len(indx)):
        res += (params[with_ind] + params[indx[i]])
    return res


def calc_total_coefs(alphs, params, interaction_info, n):
    arrays = []
    for i in range(n):
        coop_coef = alphs[i, 0]
        conflict_coef = alphs[i, 1]
        indep_coef = alphs[i, 2]

        res = coop_coef * calc(i, interaction_info["Кооперация"][i], params) \
              + conflict_coef * calc(i, interaction_info["Конфликт"][i], params) \
              + indep_coef * calc(i, interaction_info["Независимость"][i], params)
        arrays.append(res)

    combined_array = np.vstack(arrays)
    return np.sum(combined_array, axis=0)


# print(interactions / n)
# print(interactions_list)


final_coefs = calc_total_coefs(interactions / n, C, interactions_list, n)
print(final_coefs)

# C = np.array([[4, 2],
#                    [2, 4],
#                    [3, 9],
#                    [8, 2],
#                    [4, -1],
#                    [3, -2],
#                    [-2, 4],
#                    [-3, 1],
#                    [-4, -3]])
#
# A = np.array([[2, 3],
#               [-1, 3],
#               [2, -1]])
#
# b = np.array([18, 9, 10])
#
# rk = RankingSolution(A, b, C, "max")
# print(rk.solve_tasks())


# # Значения целевых функций
# fs = [28, 22, 45, 52, 22, 15, 12, 3, 0]
#
# # Матрица взаимодействия K
# K = np.array([
#     [0.8, 0.71, 0.98, 0.25, 0.49, 0.00, -0.71, -0.99, 0.8],
#     [0.99, 0.65, 0.22, -0.12, 0.60, -0.14, -0.89, 0.71, 0.99],
#     [0.54, 0.08, -0.26, 0.71, 0.00, -0.82, 0.98, 0.65, 0.54],
#     [0.88, 0.67, -0.22, -0.84, -0.97, 0.25, 0.22, 0.08, 0.88],
#     [0.94, -0.65, -0.99, -0.63, 0.49, -0.12, -0.26, 0.67, 0.94],
#     [-0.87, -0.96, -0.33, 0.00, 0.60, 0.71, -0.22, -0.65, -0.87],
#     [0.71, -0.18, -0.71, -0.14, 0.00, -0.84, -0.99, -0.96, 0.71],
#     [0.57, -0.99, -0.89, -0.82, -0.97, -0.63, -0.33, -0.18, 0.57]
# ])
#
# Матрица коэффициентов значимости alpha
# alpha = np.array([
#     [4 / 9, 5 / 9, 5 / 9, 6 / 9, 3 / 9, 3 / 9, 4 / 9, 3 / 9, 2 / 9],
#     [2 / 9, 1 / 9, 1 / 9, 2 / 9, 3 / 9, 2 / 9, 2 / 9, 4 / 9, 5 / 9],
#     [3 / 9, 3 / 9, 3 / 9, 1 / 9, 3 / 9, 4 / 9, 3 / 9, 2 / 9, 2 / 9],
# ])
#
# params = np.array([[4, 2],
#                    [2, 4],
#                    [3, 9],
#                    [8, 2],
#                    [4, -1],
#                    [3, -2],
#                    [-2, 4],
#                    [-3, 1],
#                    [-4, -3]])
#
# array1 = alpha.T
#
# print(array1 == interactions / n)
#
#
# def calc(with_ind, indx):
#     res = params[with_ind] * params[indx[0]]
#     for i in range(1, len(indx)):
#         res += (params[with_ind] + params[indx[i]])
#     return res
#
#
# array2 = np.array([[array1[0, 0] * calc(0, [0, 1, 2, 3]), array1[0, 1] * calc(0, [7, 8]), array1[0,
#                                                                                                  2] * calc(0,
#                                                                                                            [4, 5, 6])],
#                    [array1[1, 0] * calc(1, [0, 1, 2, 3, 6]), array1[1, 1] * calc(1, [1, 8]), array1[1,
#                                                                                                     2] * calc(1, [4, 5,
#                                                                                                                   7])],
#                    [array1[2, 0] * calc(2, [0, 1, 2, 3, 6]), array1[2, 1] * calc(2, [2, 8]), array1[2,
#                                                                                                     2] * calc(2, [4, 5,
#                                                                                                                   7])],
#                    [array1[3, 0] * calc(3, [0, 1, 2, 3, 4, 5]), array1[3, 1] * calc(3, [7, 8]), array1[3,
#                                                                                                        2] * calc(3,
#                                                                                                                  [6])],
#                    [array1[4, 0] * calc(4, [3, 4, 5]), array1[4, 1] * calc(4, [7, 8]), array1[4, 2] * calc(4, [0,
#                                                                                                                1, 2,
#                                                                                                                6])],
#                    [array1[5, 0] * calc(5, [3, 4, 5]), array1[5, 1] * calc(5, [6, 7]), array1[5, 2] * calc(5, [0, 1, 2,
#                                                                                                                8])],
#                    [array1[6, 0] * calc(6, [1, 2, 6, 7]), array1[6, 1] * calc(6, [5, 8]),
#                     array1[6, 2] * calc(6, [0, 3, 4])],
#                    [array1[7, 0] * calc(7, [6, 7, 8]), array1[7, 1] * calc(7, [0, 3, 4, 5]),
#                     array1[7, 2] * calc(7, [1, 3])],
#                    [array1[8, 0] * calc(8, [7, 8]), array1[8, 1] * calc(8, [0, 1, 2, 3, 4, 6]),
#                     array1[8, 2] * calc(8, [5])]])
#
# r = np.sum(array2, axis=1)
# final_result = np.sum(r, axis=0)
#
# print("Итоговый массив:", final_result)
#
# A = np.array([[2, 3],
#              [-1, 3],
#              [2, -1]])
# # C = np.array([[117.86, 103.072]])
# C_new = np.array([[final_result[0], final_result[1]]])
# b = np.array([18, 9, 10])
#
# opt = LinearOptimization(A, b, C_new, "max")
# r, v = opt.solve_cpu()
# print(r, v)

# # Умножение: элемент-wise для соответствующих элементов
# result = array1[:, :, np.newaxis] * array2  # Преобразуем array1 для элемент-wise умножения
#
# # Сложение по оси 1
# sum_per_element = np.sum(result, axis=1)  # Теперь имеет размер (9, 2)
#
# # Сложение всех 9 элементов до итогового массива размером (2,)
# final_result = np.sum(sum_per_element, axis=0)
#
# print("Итоговый массив:", final_result)
#
# A = np.array([[2, 3],
#              [-1, 3],
#              [2, -1]])
# # C = np.array([[117.86, 103.072]])
# C = np.array([[76.22222222, 64.12345679]])
# b = np.array([18, 9, 10])
# #
# res, vars = solve_problem(A, b, C, None, None)
# print(res)
# print(vars)


# df = pd.read_csv("/Users/ilabelozerov/Downloads/Housing.csv")
#
# price = df['price'].values
# area = df['area'].values
#
# data = df.loc[:, ['area', 'price']].values
# array = np.arange(df.shape[0])
# rules = array.reshape(df.shape[0], 1)
#
#
# opt = AntOptimization(
#     data=data,
#     k=5,
#     q=0.8,
#     epsilon=0.005,
#     n_iter=100,
#     ranges=[FuzzyBounds(start=data[:, 0].min() - 20, step=10, end=np.max(data[:, 0]), x="x_1")],
#     r=data[:, -1],
#     n_terms=data.shape[0],
#     n_ant=55,
#     mf_type="triangular",
#     base_rules_ind=rules
#
# )
# _ = opt.continuous_ant_algorithm()
#
# param = opt.best_result.params
#
# x = Domain((np.min(data[:, 0]) - 20, np.max(data[:, 0]), 10), name='x')
# for i in range(data.shape[0]):
#     x.create_number('triangular', *param[0, i, :].tolist(), name=f"x_{i}")
#
# rul = [
#     BaseRule(antecedents=[('x', f'x_{i}')], consequent=data[:, -1][i])
#     for i in range(data.shape[0])
# ]
#
# inference_system = SingletonInference(domains={
#     'x': x,
# }, rules=rul)
#
#
# def f(y, y_pred, m):
#     return np.sqrt(
#         np.sum(
#             np.square(y - y_pred)
#         )
#     ) / m
#
#
# res = []
# for i in range(data.shape[0]):
#     input_data = {'x': data[:, 0][i]}
#     result = inference_system.compute(input_data)
#     res.append(result)
#     print(result, data[:, -1][i])
