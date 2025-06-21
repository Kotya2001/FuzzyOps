from fuzzyops.fuzzy_optimization import LinearOptimization

import numpy as np


# setting coefficients for objective functions
C = np.array([
    [4, 2],
])


# the matrix of coefficients of constraints
A = np.array([[2, 3],
              [-1, 3],
              [2, -1]])
# vector of bounds
b = np.array([18, 9, 10])

# Solving the optomization problem
opt = LinearOptimization(A, b, C, "max")
# we get optimal values
_, v = opt.solve_cpu()
print(v)