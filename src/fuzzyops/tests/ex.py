import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_numbers import Domain

d = Domain((0, 101), name='d', method='minimax')
d.create_number('gauss', 1, 0, name='out')
for i in range(50):
    d.create_number('gauss', 1, i, name='n' + str(i))
    d.out += d.get('n' + str(i))

d.to('cpu')
print(d.out)
print(d.n49)
print(d.out.values)

# --------------------------------------------------

# from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference

# age_domain = Domain((0, 100), name='age')
# age_domain.create_number('trapezoidal', -1, 0, 20, 30, name='young')
# age_domain.create_number('trapezoidal', 20, 30, 50, 60, name='middle')
# age_domain.create_number('trapezoidal', 50, 60, 100, 100, name='old')

# accident_domain = Domain((0, 1, 0.1), name='accident')
# accident_domain.create_number('trapezoidal', -0.1, 0., 0.1, 0.2, name='low')
# accident_domain.create_number('trapezoidal', 0.1, 0.2, 0.7, 0.8, name='medium')
# accident_domain.create_number('trapezoidal', 0.7, 0.8, 0.9, 1, name='high')

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

# fuzzy_inference = FuzzyInference(domains={
#             'age': age_domain,
#             'accident': accident_domain,
#         }, rules=ruleset)

# minimax_out = fuzzy_inference.compute({"age": 25})
# print(minimax_out['accident'])


# --------------------------------------------------


# import sys
# import os
# from pathlib import Path

# root_path = Path(os.path.abspath(__file__))
# src_dir = root_path.parents[2]
# sys.path.append(src_dir.__str__())

# import numpy as np
# from fuzzyops.fuzzy_optimization import LinearOptimization

# C = np.array([[4, 2]])
# b = np.array([18, 9, 10])
# A = np.array([[2, 3], [-1, 3], [2, -1]])

# opt = LinearOptimization(A, b, C, 'max')
# r, v = opt.solve_cpu()
# print(r, v)


# --------------------------------------------------
# import numpy as np
# from fuzzyops.fuzzy_optimization import AntOptimization, FuzzyBounds

# x = np.arange(start=0.01, stop=1, step=0.01)
# r = np.array([9.919, -6.175, 4.372, -3.680, 2.663, -2.227,
#                            1.742, -2.789, 11.851, -8.565, 0.938, -0.103])
# size = r.shape[0]
# X = np.random.choice(x, size=size)
# X = np.reshape(X, (size, 1))
# data = np.hstack((X, np.reshape(r, (size, 1))))

# array = np.arange(12)
# rules = array.reshape(12, 1)

# opt = AntOptimization(
#     data=data,
#     k=5,
#     q=0.8,
#     epsilon=0.005,
#     n_iter=100,
#     ranges=[FuzzyBounds(start=0.01, step=0.01, end=1, x="x_1")],
#     r=r,
#     n_terms=12,
#     n_ant=55,
#     mf_type="triangular",
#     base_rules_ind=rules
# )
# _ = opt.continuous_ant_algorithm()
# print(opt.best_result)

# --------------------------------------------------

# from fuzzyops.fuzzy_nn import Model, process_csv_data
# import torch

# path = "/Users/ilabelozerov/FuzzyOps/src/fuzzyops/tests/Iris.csv"

# n_features = 2
# n_terms = [5, 5]
# n_out_vars = 3
# lr = 3e-4
# task_type = "classification"
# batch_size = 8
# member_func_type = "gauss"
# epochs = 100
# verbose = True
# device = 'cpu'

# X, y = process_csv_data(path=path,
#                         target_col="Species",
#                         n_features=n_features,
#                         use_label_encoder=True,
#                         drop_index=True)

# model = Model(X, y,
#               n_terms, n_out_vars,
#               lr, task_type,
#               batch_size, member_func_type,
#               epochs, verbose,
#               device=device)

# m = model.train()
# best_score = max(model.scores)
# res = m(torch.Tensor([[7, 3.2]]))
# print(res)
# print(torch.argmax(res, dim=1))
