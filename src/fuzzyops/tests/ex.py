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


temp_domain = Domain((0, 30), name='temp')
temp_domain.create_number('triangular', 0, 5, 10, name='low')
temp_domain.create_number('triangular', 10, 15, 20, name='middle')
temp_domain.create_number('triangular', 15, 25, 30, name='high')

temp_domain.plot()

heat_domain = Domain((0, 31, 1), name='heat')
heat_domain.create_number('singleton', 0, name='low')
heat_domain.create_number('singleton', 15, name='middle')
heat_domain.create_number('singleton', 30, name='high')


rules = [
    BaseRule(
        antecedents=[('temp', 'low')],
        consequent=('heat_domain', 'high'),
    ),
    BaseRule(
        antecedents=[('temp', 'middle')],
        consequent=('heat_domain', 'middle'),
    ),
    BaseRule(
        antecedents=[('temp', 'high')],
        consequent=('heat_domain', 'low'),
    ),
]

inference_system = FuzzyInference(domains={
    'temp': temp_domain,
    'heat_domain': heat_domain,
}, rules=rules)

input_data = {
    'temp': 1,
}

result = inference_system.compute(input_data)
print(result)

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



