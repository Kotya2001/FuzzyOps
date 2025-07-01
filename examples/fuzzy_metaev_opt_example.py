"""
The metaheuristic optimization algorithm is suitable for approximating any functions,
from one or more variables,
the algorithm finds the parameters of the membership functions for a singleton rule base (the output is given by a clear value)
next, you need to create a rule base to input dependent variables (x) and obtain outputs (y)

 Consider the following task:
 It is necessary to automatically construct and find the parameters of the membership functions for approximating the following data
 from the website https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?select=Housing.csv
 about the prices of the area and other features of the houses
 and approximate the dependence of the area of the house on the feature of the price of the house.
 Input variables - the area of the house
 Output price of the house
"""

from fuzzyops.fuzzy_optimization import AntOptimization, FuzzyBounds
from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.fuzzy_logic import BaseRule, SingletonInference

import numpy as np
import pandas as pd

df = pd.read_csv("Housing.csv")

price = df['price'].values
area = df['area'].values

data = df.loc[:, ['area', 'price']].values

# these variables specify the terms for each rule
# (this results in a system with 1 variable and 1 output, with a dimension of data.shape[0] by 1
array = np.arange(df.shape[0])
rules = array.reshape(df.shape[0], 1)

# Let's set the algorithm parameters
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

# Building a domain region for an input variable
x = Domain((np.min(data[:, 0]) - 20, np.max(data[:, 0]), 10), name='x')
for i in range(data.shape[0]):
    x.create_number('triangular', *param[0, i, :].tolist(), name=f"x_{i}")

# building a Singleton-type rule base
rul = [
    BaseRule(antecedents=[('x', f'x_{i}')], consequent=data[:, -1][i])
    for i in range(data.shape[0])
]

inference_system = SingletonInference(domains={
    'x': x,
}, rules=rul)

# We feed data to the input and get the result
input_data = {'x': 8000}
result = inference_system.compute(input_data)
print(result)