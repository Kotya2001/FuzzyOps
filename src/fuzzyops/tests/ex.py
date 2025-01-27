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

from fuzzyops.fuzzy_optimization import solve_problem

import numpy as np
import pandas as pd
from sympy import symbols, diff, expand

# Значения целевых функций
fs = [28, 22, 45, 52, 22, 15, 12, 3, 0]

# Матрица взаимодействия K
K = np.array([
    [0.8, 0.71, 0.98, 0.25, 0.49, 0.00, -0.71, -0.99, 0.8],
    [0.99, 0.65, 0.22, -0.12, 0.60, -0.14, -0.89, 0.71, 0.99],
    [0.54, 0.08, -0.26, 0.71, 0.00, -0.82, 0.98, 0.65, 0.54],
    [0.88, 0.67, -0.22, -0.84, -0.97, 0.25, 0.22, 0.08, 0.88],
    [0.94, -0.65, -0.99, -0.63, 0.49, -0.12, -0.26, 0.67, 0.94],
    [-0.87, -0.96, -0.33, 0.00, 0.60, 0.71, -0.22, -0.65, -0.87],
    [0.71, -0.18, -0.71, -0.14, 0.00, -0.84, -0.99, -0.96, 0.71],
    [0.57, -0.99, -0.89, -0.82, -0.97, -0.63, -0.33, -0.18, 0.57]
])

# Матрица коэффициентов значимости alpha
alpha = np.array([
    [4 / 9, 5 / 9, 5 / 9, 6 / 9, 3 / 9, 3 / 9, 4 / 9, 3 / 9, 2 / 9],
    [2 / 9, 1 / 9, 1 / 9, 2 / 9, 3 / 9, 2 / 9, 2 / 9, 4 / 9, 5 / 9],
    [3 / 9, 3 / 9, 3 / 9, 1 / 9, 3 / 9, 4 / 9, 3 / 9, 2 / 9, 2 / 9],
])

params = np.array([[4, 2],
                   [2, 4],
                   [3, 9],
                   [8, 2],
                   [4, -1],
                   [3, -2],
                   [-2, 4],
                   [-3, 1],
                   [-4, -3]])

array1 = alpha.T


def calc(with_ind, indx):
    res = params[with_ind] * params[indx[0]]
    for i in range(1, len(indx)):
        res += (params[with_ind] + params[indx[i]])
    return res


array2 = np.array([[array1[0, 0] * calc(0, [0, 1, 2, 3]), array1[0, 1] * calc(0, [7, 8]), array1[0,
                                                                                                 2] * calc(0,
                                                                                                           [4, 5, 6])],
                   [array1[1, 0] * calc(1, [0, 1, 2, 3, 6]), array1[1, 1] * calc(1, [1, 8]), array1[1,
                                                                                                    2] * calc(1, [4, 5,
                                                                                                                  7])],
                   [array1[2, 0] * calc(2, [0, 1, 2, 3, 6]), array1[2, 1] * calc(2, [2, 8]), array1[2,
                                                                                                    2] * calc(2, [4, 5,
                                                                                                                  7])],
                   [array1[3, 0] * calc(3, [0, 1, 2, 3, 4, 5]), array1[3, 1] * calc(3, [7, 8]), array1[3,
                                                                                                       2] * calc(3,
                                                                                                                 [6])],
                   [array1[4, 0] * calc(4, [3, 4, 5]), array1[4, 1] * calc(4, [7, 8]), array1[4, 2] * calc(4, [0,
                                                                                                               1, 2,
                                                                                                               6])],
                   [array1[5, 0] * calc(5, [3, 4, 5]), array1[5, 1] * calc(5, [6, 7]), array1[5, 2] * calc(5, [0, 1, 2,
                                                                                                               8])],
                   [array1[6, 0] * calc(6, [1, 2, 6, 7]), array1[6, 1] * calc(6, [5, 8]),
                    array1[6, 2] * calc(6, [0, 3, 4])],
                   [array1[7, 0] * calc(7, [6, 7, 8]), array1[7, 1] * calc(7, [0, 3, 4, 5]),
                    array1[7, 2] * calc(7, [1, 3])],
                   [array1[8, 0] * calc(8, [7, 8]), array1[8, 1] * calc(8, [0, 1, 2, 3, 4, 6]),
                    array1[8, 2] * calc(8, [5])]])

# Умножение: элемент-wise для соответствующих элементов
result = array1[:, :, np.newaxis] * array2  # Преобразуем array1 для элемент-wise умножения

# Сложение по оси 1
sum_per_element = np.sum(result, axis=1)  # Теперь имеет размер (9, 2)

# Сложение всех 9 элементов до итогового массива размером (2,)
final_result = np.sum(sum_per_element, axis=0)

print("Итоговый массив:", final_result)

A = np.array([[2, 3],
             [-1, 3],
             [2, -1]])
# C = np.array([[117.86, 103.072]])
C = np.array([[76.22222222, 64.12345679]])
b = np.array([18, 9, 10])
#
res, vars = solve_problem(A, b, C, None, None)
print(res)
print(vars)


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
