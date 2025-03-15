from fuzzyops.fuzzy_optimization import LinearOptimization

import numpy as np


# задачем коэффициента при целевых функицях
C = np.array([
    [4, 2],
])


# матрица коэффициентов ограничений
A = np.array([[2, 3],
              [-1, 3],
              [2, -1]])

b = np.array([18, 9, 10])

# Рещаем задачу оптмизации
opt = LinearOptimization(A, b, C, "max")
# получаем оптимальные значения
_, v = opt.solve_cpu()
print(v)