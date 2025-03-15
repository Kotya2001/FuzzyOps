import numpy as np
from fuzzyops.fuzzy_optimization import LinearOptimization

C = np.array([[4, 2]])
b = np.array([18, 9, 10])
A = np.array([[2, 3], [-1, 3], [2, -1]])

opt = LinearOptimization(A, b, C, 'max')
r, v = opt.solve_cpu()
print(r, v)