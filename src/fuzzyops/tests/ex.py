"""
Задача:
    Упарвление некоторой фирмой по производству какой-либо продукции (например мебель)
    Необходимо увеличить производстов, но при этом есть некоторые ограничения:
        ограничение на повышение производительности труда,
        ограничение на улучшение качетсва продукции,
        ограничение на время проведения маркетинговых усилий


    Допустим, мы определили 5 ключевых показателей эффективности, которые хотим повысить

    1.Производительность труда (f1) - увеличение количества выпускаемой продукции за час;
    2.Качество продукции (f2) - Уменьшение числа дефектов на единицу продукции;
    3.Удовлетворенность клиентов (f3) - Повышение уровня удовлетворенности клиентов благодаря улучшению качества;
    4.Эффективность использования оборудования (f4) - минимизация времени простоя оборудования;
    5.Маркетинговые усилия (f5) - Увеличение доли рынка засчет продвижения продукции.

    При этом определим искомые переменные (x1, x2):

        x1 - количество часов, которое выделяется на повышение производительности труда (f1),
        и соответственно мининимазию времени простоя оборудования (f4),
        x2 - может обозначать количество часов, которое выделяется (f2) на улучшение качества продукции,
        удовлетворенности клиентов клиентов (f3), и маркетинговых усислий (f5)

    То есть задачи принимает следующий вид:

        f1(x) = A11 * x1 + B12 * x2 -> max;
        f2(x) = A21 * x1 + B22 * x2 -> max;
        f3(x) = A31 * x1 + B32 * x2 -> max;
        f4(x) = A41 * x1 + B42 * x2 -> max;
        f5(x) = A51 * x1 + B52 * x2 -> max;

    При этом коэффициенты при x1 и x2 в каждом выржении являются нечеткими числами типа LR (теругольное и унимодальное),
    точнее коэффициентами для задания нечеткого числа такого типа

    Например, для данной задачи коэффициенты будут выглядеть так:
        [
            [[4, 1, 7], [2, 0, 5]],
            [[2, 1, 3], [4, 2, 6]],
            [[3, -1, 4], [9, 5, 12]],
            [[4, 3, 8], [-1, -2, 1]],
            [[3, 1, 6], [-2, -4, 2]],
        ]

        То есть размерность (6, 2, 3), где 6 - это число функций, 2 - чило переменных,
         3 - параметры для задания теругольного числа

         Стоит отметить, что параметры задаются в таком порядке [модальное значени, левая граница, правая граница]
         [4, 1, 7], 4 - модальное значение, 1 - левая граница, 7 -правая граница

    Также задаются ограничения для задачи оптимизации:

        1) 2 * x1 + 2 * x2 <= 8;

            Это ограничение говорит о том, что общее количество часов,
            можно выделить на повышение производительности труда (x1)
            и улучшение качества продукции/удовлетворенности клиентов/маркетинговых усилий (x2)
            не должно превышать 8 часов в день.

        2) -x1 + 3 * x2 <= 9;

            Это ограничение указывает на то, что количество часов,
            можно потратить на улучшение качества продукции/удовлетворенности клиентов/маркетинговых усилий (х2)
            не должно превышать 9 часов в день, при условии,
            что тратится минимум 1 час на повышение производительности труда (х1).

        3) 2 * x1 - x2 <= 10;

            Это ограничение показывает что сумма времени, которая тратится на минимизацию простое ообрудования (х1)
            и маркетинговых усилий (х2) не должна превышать 10 часов в день.

        4) x1 >= 0, x2 >= 0; Кол-во часов не должно быть отрицательным
"""

import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_optimization import solve_problem, \
    LinearOptimization, calc_total_functions, get_interaction_matrix, check_LR_type, calc_total_functions
from fuzzyops.fuzzy_numbers import Domain

import numpy as np

coefs_domain = Domain((-20, 50, 1), name='coefs')
n = 5

# задачем нечеткие коэффициента при целевых функицях, для определения коэффициентов взаимодейтсвия
f_num_params = np.array([
    [[4, 1, 7], [2, 0, 5]],
    [[2, 1, 3], [4, 2, 6]],
    [[3, -1, 4], [9, 5, 12]],
    [[4, 3, 8], [-1, -2, 1]],
    [[3, 1, 6], [-2, -4, 2]],
])

# берем модальные значения
C = f_num_params[:, :, 0]

C_f = []

for i in range(len(f_num_params)):
    lst = []
    for j in range(len(f_num_params[i])):
        coefs = [f_num_params[i][j].tolist()[1], f_num_params[i][j].tolist()[0], f_num_params[i][j].tolist()[2]]
        lst.append(coefs_domain.create_number('triangular', *coefs, name=f"c{i}{j}"))
    C_f.append(np.array(lst))

C_f = np.array(C_f)

# проверяем соответствие нечетких чисел с нашмим коэффициентами, что соответствуют LR-типу
assert check_LR_type(C_f)

A = np.array([[2, 3],
              [-1, 3],
              [2, -1]])

b = np.array([18, 9, 10])

# Находим коэффициенты и таблицу как функции соотносятся друг с другом (Кооперируют, Конфликтуют, Независят)
alphas, interactions_list = get_interaction_matrix(f_num_params)
# Строим коэффициенты при переменных у обобщенной целевой функции по алгоритму
final_coefs = calc_total_functions(alphas, C, interactions_list, n)

C_new = np.array([[final_coefs[0], final_coefs[1]]])

# Рещаем задачу оптмизации
opt = LinearOptimization(A, b, C_new, "max")
_, v = opt.solve_cpu()
print(v) # [6, 2]

# k = np.array([[1, 0.8, 0.71, 0.98, 0.25, 0.49, 0, -0.71, -0.99],
#               [0.8, 1, 0.99, 0.65, 0.22, -0.12, 0.6, -0.14, -0.89],
#               [0.71, 0.99, 1, 0.54, 0.08, -0.26, 0.71, 0, -0.82],
#               [0.98, 0.65, 0.54, 1, 0.88, 0.67, -0.22, -0.84, -0.97],
#               [0.25, 0.22, 0.08, 0.88, 1, 0.94, -0.65, -0.99, -0.63],
#               [0.49, -0.12, -0.26, 0.67, 0.94, 1, -0.87, -0.96, -0.33],
#               [0, 0.6, 0.71, -0.22, -0.65, -0.87, 1, 0.71, -0.18],
#               [-0.71, -0.14, 0, -0.84, -0.99, -0.96, 0.71, 1, 0.57],
#               [-0.99, -0.89, -0.82, -0.97, -0.63, -0.33, -0.18, 0.57, 1]])
#
# C = np.array([[4, 2],
#               [2, 4],
#               [3, 9],
#               [8, 2],
#               [4, -1],
#               [3, -2],
#               [-2, 4],
#               [-3, 1],
#               [-4, -3]])

# n = 9
# interactions = np.zeros((n, 3))
# total_info = {"Кооперация": [[] for _ in range(n)],
#               "Конфликт": [[] for _ in range(n)],
#               "Независимость": [[] for _ in range(n)]}
#
#
# def _define_interaction_type(table: np.ndarray,
#                              k: np.ndarray,
#                              total_info: dict) -> np.ndarray:
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
#     for index, _ in np.ndenumerate(k):
#         row, col = index[0], index[1]
#
#         if 0.5 <= k[row][col] <= 1:
#             table[row][0] += 1
#             total_info["Кооперация"][row].append(col)
#         elif -1 <= k[row][col] <= -0.5:
#             table[row][1] += 1
#             total_info["Конфликт"][row].append(col)
#         elif -0.5 < k[row][col] < 0.5:
#             table[row][2] += 1
#             total_info["Независимость"][row].append(col)
#
#     return table, total_info
#
#
# interactions, interactions_list = _define_interaction_type(interactions, k, total_info)
#
#
# def calc(with_ind, indx, params):
#     res = params[with_ind] * params[indx[0]]
#     for i in range(1, len(indx)):
#         res += (params[with_ind] + params[indx[i]])
#     return res
#
#
# def calc_total_coefs(alphs, params, interaction_info, n):
#     arrays = []
#     for i in range(n):
#         coop_coef = alphs[i, 0]
#         conflict_coef = alphs[i, 1]
#         indep_coef = alphs[i, 2]
#
#         res = coop_coef * calc(i, interaction_info["Кооперация"][i], params) \
#               + conflict_coef * calc(i, interaction_info["Конфликт"][i], params) \
#               + indep_coef * calc(i, interaction_info["Независимость"][i], params)
#         arrays.append(res)
#
#     combined_array = np.vstack(arrays)
#     return np.sum(combined_array, axis=0)


# print(interactions / n)
# print(interactions_list)


# final_coefs = calc_total_coefs(interactions / n, C, interactions_list, n)
# print(final_coefs)

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
