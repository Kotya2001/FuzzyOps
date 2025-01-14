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

from fuzzyops.fuzzy_optimization import AntOptimization, FuzzyBounds
from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber
from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference

import numpy as np
import pandas as pd

# temp_domain = Domain((0, 30), name='temp')
# temp_domain.create_number('triangular', 0, 5, 10, name='low')
# temp_domain.create_number('triangular', 10, 15, 20, name='middle')
# temp_domain.create_number('triangular', 15, 25, 30, name='high')
#
# heat_domain = Domain((0, 31, 1), name='heat')
# heat_domain.create_number('singleton', 0, name='low')
# heat_domain.create_number('singleton', 15, name='middle')
# heat_domain.create_number('singleton', 30, name='high')
#
# rules = [
#     BaseRule(
#         antecedents=[('temp', 'low')],
#         consequent=('heat_domain', 'high'),
#     ),
#     BaseRule(
#         antecedents=[('temp', 'middle')],
#         consequent=('heat_domain', 'middle'),
#     ),
#     BaseRule(
#         antecedents=[('temp', 'high')],
#         consequent=('heat_domain', 'low'),
#     ),
# ]


df = pd.read_csv("/Users/ilabelozerov/FuzzyOps/src/fuzzyops/tests/sales.csv")
# df = df.iloc[:, :-1]
r = np.array([600, 200, 200, 400])
data = df.values
X = df.iloc[:, :-1]
rules = np.array([[0, 1],
                  [1, 0],
                  [2, 2],
                  [2, 2]])

x1 = Domain((500, 5700, 10), name='x_1')
x1.create_number('triangular', 1751.09833929, 3522.03918129, 4206.59217367, name='x_1_1')
x1.create_number('triangular', 1311.49299969, 2700.2685087, 4119.56733172, name='x_1_2')
x1.create_number('triangular', 661.93233222, 5261.62975796, 5465.64812383, name='x_1_3')

x2 = Domain((20, 120, 1), name='x_2')
x2.create_number('triangular', 77.44490918,   97.19210937,  112.70155828, name='x_2_1')
x2.create_number('triangular', 37.45437608,   62.51049707,  114.07065485, name='x_2_2')
x2.create_number('triangular', 47.86961377,   55.26528166,  109.14513497, name='x_2_3')

y = Domain((100, 650, 10), name='y')
y.create_number('singleton', 600, name='y1')
y.create_number('singleton', 200, name='y2')
y.create_number('singleton', 200, name='y3')
y.create_number('singleton', 400, name='y4')

rul = [
    BaseRule(
        antecedents=[('x_1', 'x_1_1'), ('x_2', 'x_2_1')],
        consequent=('y', 'y1'),
    ),
    BaseRule(
        antecedents=[('x_1', 'x_1_2'), ('x_2', 'x_2_1')],
        consequent=('y', 'y2'),
    ),
    BaseRule(
        antecedents=[('x_1', 'x_1_3'), ('x_2', 'x_2_3')],
        consequent=('y', 'y3'),
    ),
    BaseRule(
        antecedents=[('x_1', 'x_1_3'), ('x_2', 'x_2_3')],
        consequent=('y', 'y4'),
    ),
]

inference_system = FuzzyInference(domains={
    'x_1': x1,
    'x_2': x2,
    'y': y
}, rules=rul)

input_data = {
    'x_1': 5000,
    'x_2': 59
}

result = inference_system.compute(input_data)
print(result)

# print(data)

# def __generate_theta():
#     theta = np.zeros((5, 2, 3, 3))
#     for j in range(2):  # Iterate over columns
#         col_name = X.columns[j]
#         low = X[col_name].min()
#         high = X[col_name].max()
#
#         theta[:, j, :, :] = np.random.uniform(low=low, high=high, size=(5,
#                                                                         3,
#                                                                         3))
#     return theta
#
#
# th = __generate_theta()
# th.sort()
# print(th)

# theta = np.random.uniform(low=700,
#                           high=5000,
#                           size=(5, 2, 3, 3))

# th = [f"th_{i}" for i in range(18)]
# ex = pd.DataFrame({key: np.zeros((5,)) for key in th + ["w", "loss"]})
# ranges = [FuzzyBounds(start=700, step=10, end=5200, x="x_1")]
# order = [line.x + f"_{str(i)}" for line in ranges for i in range(3)]


# print(ex)
# opt = AntOptimization(
#     data=data,
#     k=5,
#     q=0.8,
#     epsilon=0.005,
#     n_iter=100,
#     ranges=[FuzzyBounds(start=500, step=10, end=5700, x="x_1"),
#             FuzzyBounds(start=20, step=1, end=120, x="x_2")],
#     r=r,
#     n_terms=3,
#     n_ant=55,
#     mf_type="triangular",
#     base_rules_ind=rules
#
# )
# _ = opt.continuous_ant_algorithm()
# #
# print(opt.best_result.params)

# inference_system = FuzzyInference(domains={
#     'temp': temp_domain,
#     'heat_domain': heat_domain,
# }, rules=rules)
#
# input_data = {
#     'temp': 1,
# }
#
# result = inference_system.compute(input_data)
# print(result)


# x1 = np.arange(start=0, stop=30, step=1)
# x2 = np.arange(start=0, stop=24, step=1)
#
# attack_prob = Domain((0, 1, 0.01), name='attack')
# attack_prob.create_number('trapezoidal', -0.1, 0., 0.1, 0.3, name='low')
# attack_prob.create_number('trapezoidal', 0.3, 0.43, 0.6, 0.7, name='middle')
# attack_prob.create_number('trapezoidal', 0.65, 0.8, 0.9, 1, name='high')
#
# r = np.array([attack_prob.low.defuzz(), attack_prob.middle.defuzz(), attack_prob.high.defuzz()])
# size = r.shape[0]
#
# X1 = np.random.choice(x1, size=size)
# X1 = np.reshape(X1, (size, 1))
#
# X2 = np.random.choice(x2, size=size)
# X2 = np.reshape(X2, (size, 1))
#
# data = np.hstack((X1, X2, np.reshape(r, (size, 1))))
#
# opt = AntOptimization(
#     data=data,
#     k=5,
#     q=0.8,
#     epsilon=0.005,
#     n_iter=100,
#     ranges=[FuzzyBounds(start=0, step=1, end=30, x="x_1"),
#             FuzzyBounds(start=0, step=1, end=24, x="x_2")],
#     r=r,
#     n_terms=2,
#     n_ant=55,
#     mf_type="triangular"
# )
# _ = opt.continuous_ant_algorithm()
# print(opt.best_result)
