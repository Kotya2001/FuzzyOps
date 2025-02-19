import numpy as np
from fuzzyops.fuzzy_optimization import AntOptimization, FuzzyBounds

x = np.arange(start=0.01, stop=1, step=0.01)
r = np.array([9.919, -6.175, 4.372, -3.680, 2.663, -2.227,
                           1.742, -2.789, 11.851, -8.565, 0.938, -0.103])
size = r.shape[0]
X = np.random.choice(x, size=size)
X = np.reshape(X, (size, 1))
data = np.hstack((X, np.reshape(r, (size, 1))))

array = np.arange(12)
rules = array.reshape(12, 1)

opt = AntOptimization(
    data=data,
    k=5,
    q=0.8,
    epsilon=0.005,
    n_iter=100,
    ranges=[FuzzyBounds(start=0.01, step=0.01, end=1, x="x_1")],
    r=r,
    n_terms=12,
    n_ant=55,
    mf_type="triangular",
    base_rules_ind=rules
)
_ = opt.continuous_ant_algorithm()
print(opt.best_result)