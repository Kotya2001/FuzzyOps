import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())


"""
Регрессия с нечеткими данными методом наименьших квадратов

Задача: Оценка теплопроводности материала с учетом треугольных нечетких данных

Описание задачи:
Необходимо оценить зависимость теплопроводности λ(T) от температуры T на основе экспериментальных
данных с нечеткими погрешностями, представленными треугольными функциями принадлежности

Входные переменные - нечеткие переменные измеренной температуры и соответствующие нечеткие переменные измеренной теплопроводности
Выходные - коэффициенты функции теплопроводности a и b, где a - угловой коэффициент, b - свободный член, и RMSE решения
"""

from fuzzyops.prediction import fit_fuzzy_linear_regression, convert_fuzzy_number_for_lreg
from fuzzyops.fuzzy_numbers import Domain

temp_domain = Domain((0, 110, 0.01), name='Temperature')
# Четкие числа записываем в виде треугольных нечетких чисел без хвостов
temp_values = [
    temp_domain.create_number('triangular', 18, 20, 22),
    temp_domain.create_number('triangular', 38, 40, 42),
    temp_domain.create_number('triangular', 58, 60, 62),
    temp_domain.create_number('triangular', 78, 80, 82),
    temp_domain.create_number('triangular', 98, 100, 102)
]

tran_domain = Domain((1, 2, 0.01), name="Transcalency")
tran_values = [
    tran_domain.create_number('triangular', 1.2, 1.25, 1.3),
    tran_domain.create_number('triangular', 1.28, 1.35, 1.42),
    tran_domain.create_number('triangular', 1.35, 1.45, 1.55),
    tran_domain.create_number('triangular', 1.5, 1.62, 1.74),
    tran_domain.create_number('triangular', 1.65, 1.8, 1.95)
]

# коэффициенты регресси где a - угловой коэффициент, b - свободный член, и RMSE решения
a, b, error = fit_fuzzy_linear_regression(temp_values, tran_values)
print(a, b, error)

# Значение для осущетсвления прогноза
X_test = convert_fuzzy_number_for_lreg(temp_domain.create_number('triangular', 99, 103, 107))

# Построение прогноза
Y_pred = (X_test * a) + b

print(Y_pred.to_fuzzy_number())

# ######################################################
# # 4. Определение нечеткой иерархии
# ######################################################
#
# d.create_number('triangular', 1, 5, 11, name='cw11')
# d.create_number('triangular', 3, 5, 7, name='cw12')
# d.create_number('triangular', 0, 9, 13, name='cw13')
#
# d.create_number('triangular', 4, 5, 7, name='cw21')
# d.create_number('triangular', 3, 6, 13, name='cw22')
# d.create_number('triangular', 2, 7, 11, name='cw23')
#
# d.create_number('triangular', 5, 6, 7, name='cw31')
# d.create_number('triangular', 1, 4, 7, name='cw32')
# d.create_number('triangular', 2, 7, 11, name='cw33')
#
# criteria_weights = [
#     [d.cw11, d.cw12, d.cw13],
#     [d.cw21, d.cw22, d.cw23],
#     [d.cw31, d.cw32, d.cw33],
# ]
#
# d.create_number('triangular', 3, 6, 13, name='cc11')
# d.create_number('triangular', 3, 6, 13, name='cc12')
#
# d.create_number('triangular', 5, 6, 7, name='cc21')
# d.create_number('triangular', 2, 7, 11, name='cc22')
#
# cost_comparisons = [
#     [d.cc11, d.cc12],
#     [d.cc21, d.cc22],
# ]
#
# d.create_number('triangular', 5, 6, 7, name='qc11')
# d.create_number('triangular', 2, 7, 11, name='qc12')
#
# d.create_number('triangular', 3, 6, 13, name='qc21')
# d.create_number('triangular', 3, 6, 13, name='qc22')
#
#
# quality_comparisons = [
#     [d.qc11, d.qc12],
#     [d.qc21, d.qc22],
# ]
#
# d.create_number('triangular', 1, 5, 11, name='rc11')
# d.create_number('triangular', 3, 5, 7, name='rc12')
#
# d.create_number('triangular', 4, 5, 7, name='rc21')
# d.create_number('triangular', 3, 6, 13, name='rc22')
#
# reliability_comparisons = [
#     [d.rc11, d.rc12],
#     [d.rc21, d.rc22],
# ]
#
# alternative_comparisons = [cost_comparisons, quality_comparisons, reliability_comparisons]
#
# # Аналитическая иерархия
# hierarchy_result = fuzzy_hierarchy_solver(criteria_weights, alternative_comparisons)
# print("Нечеткая аналитическая иерархия:", hierarchy_result)

# import numpy as np
# from fuzzyops.fuzzy_optimization import LinearOptimization
#
# C = np.array([[4, 2]])
# b = np.array([18, 9, 10])
# A = np.array([[2, 3], [-1, 3], [2, -1]])
#
# opt = LinearOptimization(A, b, C, 'max')
# r, v = opt.solve_cpu()
# print(r, v)
# from fuzzyops.fuzzy_numbers import Domain
#
# # d = Domain((0, 101), name='d', method='minimax')
# # d.create_number('gauss', 1, 0, name='out')
# # for i in range(50):
# #     d.create_number('gauss', 1, i, name='n' + str(i))
# #     d.out += d.get('n' + str(i))
#
# # d.to('cpu')
# # print(d.out)
# # print(d.n49)
# # print(d.out.values)
#
# # --------------------------------------------------
#
# from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference
#
# age_domain = Domain((0, 100), name='age')
# age_domain.create_number('trapezoidal', -1, 0, 20, 30, name='young')
# age_domain.create_number('trapezoidal', 20, 30, 50, 60, name='middle')
# age_domain.create_number('trapezoidal', 50, 60, 100, 100, name='old')
#
# accident_domain = Domain((0, 1, 0.1), name='accident')
# accident_domain.create_number('trapezoidal', -0.1, 0., 0.1, 0.2, name='low')
# accident_domain.create_number('trapezoidal', 0.1, 0.2, 0.7, 0.8, name='medium')
# accident_domain.create_number('trapezoidal', 0.7, 0.8, 0.9, 1, name='high')
#
# ruleset = [
#             BaseRule(
#                 antecedents=[('age', 'young')],
#                 consequent=('accident', 'high'),
#             ),
#             BaseRule(
#                 antecedents=[('age', 'middle')],
#                 consequent=('accident', 'medium'),
#             ),
#             BaseRule(
#                 antecedents=[('age', 'old')],
#                 consequent=('accident', 'high'),
#             ),
#         ]
#
# fuzzy_inference = FuzzyInference(domains={
#             'age': age_domain,
#             'accident': accident_domain,
#         }, rules=ruleset)
#
# result = fuzzy_inference.compute({"age": 25})
# print(result['accident'].__float__())
# # print(minimax_out['accident'], type(minimax_out['accident']))
#
#
# # --------------------------------------------------
#
#
# # import sys
# # import os
# # from pathlib import Path
#
# # root_path = Path(os.path.abspath(__file__))
# # src_dir = root_path.parents[2]
# # sys.path.append(src_dir.__str__())
#
# # import numpy as np
# # from fuzzyops.fuzzy_optimization import LinearOptimization
#
# # # C (значения для критериев) должны быть 2D
# # C = np.array([[4, 2]])
# # b = np.array([18, 9, 10])
# # A = np.array([[2, 3], [-1, 3], [2, -1]])
#
# # opt = LinearOptimization(A, b, C, 'max')
# # r, v = opt.solve_cpu()
# # print(r, v)
#
# # --------------------------------------------------
#
# # from fuzzyops.fuzzy_nn import Model, process_csv_data
# # import torch
#
# # path = "/Users/ilabelozerov/FuzzyOps/src/fuzzyops/tests/Iris.csv"
#
# # n_features = 2
# # n_terms = [5, 5]
# # n_out_vars = 3
# # lr = 3e-4
# # task_type = "classification"
# # batch_size = 8
# # member_func_type = "gauss"
# # epochs = 100
# # verbose = True
# # device = 'cpu'
#
# # X, y = process_csv_data(path=path,
# #                         target_col="Species",
# #                         n_features=n_features,
# #                         use_label_encoder=True,
# #                         drop_index=True)
#
# # model = Model(X, y,
# #               n_terms, n_out_vars,
# #               lr, task_type,
# #               batch_size, member_func_type,
# #               epochs, verbose,
# #               device=device)
#
# # m = model.train()
# # best_score = max(model.scores)
# # res = m(torch.Tensor([[7, 3.2]]))
# # print(res)
# # print(torch.argmax(res, dim=1))
