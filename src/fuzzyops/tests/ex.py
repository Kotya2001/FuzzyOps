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
from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference, SingletonInference

import numpy as np
import pandas as pd

df = pd.read_csv("/Users/ilabelozerov/Downloads/Housing.csv")

price = df['price'].values
area = df['area'].values

data = df.loc[:, ['area', 'price']].values
array = np.arange(df.shape[0])
rules = array.reshape(df.shape[0], 1)


opt = AntOptimization(
    data=data,
    k=5,
    q=0.8,
    epsilon=0.005,
    n_iter=100,
    ranges=[FuzzyBounds(start=data[:, 0].min() - 20, step=10, end=np.max(data[:, 0]), x="x_1")],
    r=data[:, -1],
    n_terms=data.shape[0],
    n_ant=55,
    mf_type="triangular",
    base_rules_ind=rules

)
_ = opt.continuous_ant_algorithm()

param = opt.best_result.params

x = Domain((np.min(data[:, 0]) - 20, np.max(data[:, 0]), 10), name='x')
for i in range(data.shape[0]):
    x.create_number('triangular', *param[0, i, :].tolist(), name=f"x_{i}")

rul = [
    BaseRule(antecedents=[('x', f'x_{i}')], consequent=data[:, -1][i])
    for i in range(data.shape[0])
]

inference_system = SingletonInference(domains={
    'x': x,
}, rules=rul)


def f(y, y_pred, m):
    return np.sqrt(
        np.sum(
            np.square(y - y_pred)
        )
    ) / m


res = []
for i in range(data.shape[0]):
    input_data = {'x': data[:, 0][i]}
    result = inference_system.compute(input_data)
    res.append(result)
    print(result, data[:, -1][i])


