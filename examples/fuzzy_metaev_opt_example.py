"""
Алгоритм метаэвиристической оптимизации подходит для аппроксимации каких-либо функций,
от одной или нескольких переменных,
алгоритм находит параметры функций принадлежности для базы правил типа "синглтон" (Выход задан четким значением)
далее нужно создать базу правил, чтобы подать на вход зависимые переменные (х) и получить выходы (у)

    Рассмотрим задачу:
    Необходимо построить автоматически и найти параметры функций принадлежности для аппроксимации следующих данных
    с сайта https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?select=Housing.csv
    о ценах площади и других признаках домов
    и приблизим зависимость площади дома от признакак цены дома.
    Входные переменные - площадь дома
    Выходные цена дома
"""

# (Библиотека уже установлена в ваш проект)
from fuzzyops.fuzzy_optimization import AntOptimization, FuzzyBounds
from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.fuzzy_logic import BaseRule, SingletonInference

import numpy as np
import pandas as pd

df = pd.read_csv("/Users/ilabelozerov/Downloads/Housing.csv")

price = df['price'].values
area = df['area'].values

data = df.loc[:, ['area', 'price']].values

# эти переменные указывают термы для каждого правила
# (получается системы из 1 переменной и 1 выхода размерностью data.shape[0] на 1
array = np.arange(df.shape[0])
rules = array.reshape(df.shape[0], 1)

# Зададим параметры алгоритма
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

# Строим доменную область для входной переменной
x = Domain((np.min(data[:, 0]) - 20, np.max(data[:, 0]), 10), name='x')
for i in range(data.shape[0]):
    x.create_number('triangular', *param[0, i, :].tolist(), name=f"x_{i}")

# строим базу правил типа Синглтон
rul = [
    BaseRule(antecedents=[('x', f'x_{i}')], consequent=data[:, -1][i])
    for i in range(data.shape[0])
]

inference_system = SingletonInference(domains={
    'x': x,
}, rules=rul)

# Подаем данные на вход получаем результат
input_data = {'x': 8000}
result = inference_system.compute(input_data)