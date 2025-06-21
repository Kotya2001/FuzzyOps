"""
Task:
    If a company produces some kind of product (for example, furniture), 
    it is necessary to increase production, but there are some limitations:
        limitation on increasing labor productivity,
        limitation on improving product quality,
        limitation on the time of marketing efforts


    Let's say we have identified 5 key performance indicators that we want to improve:

    1. Labor productivity (f1) - increasing the number of products produced per hour;
    2. Product quality (f2) - reducing the number of defects per product;
    3. Customer satisfaction (f3) - improving customer satisfaction by enhancing product quality;
    4. Equipment utilization efficiency (f4) - minimizing equipment downtime;
    5. Marketing efforts (f5) - Increasing market share by promoting products.

    In this case, we define the desired variables (x1, x2):

        x1 is the number of hours allocated to increase productivity (f1)
        and, accordingly, minimize equipment downtime (f4),
        x2 can denote the number of hours allocated (f2) to improve product quality,
        customer satisfaction (f3), and marketing efforts (f5).

    In other words, the tasks take the following form:

        f1(x) = A11 * x1 + B12 * x2 -> max;
        f2(x) = A21 * x1 + B22 * x2 -> max;
        f3(x) = A31 * x1 + B32 * x2 -> max;
        f4(x) = A41 * x1 + B42 * x2 -> max;
        f5(x) = A51 * x1 + B52 * x2 -> max;

    At the same time, the coefficients for x1 and x2 in each expression are fuzzy numbers of the LR type (terugol and unimodal),
    or rather, the coefficients for specifying a fuzzy number of this type

    For example, for this problem, the coefficients will look like this:
        [
            [[4, 1, 7], [2, 0, 5]],
            [[2, 1, 3], [4, 2, 6]],
            [[3, -1, 4], [9, 5, 12]],
            [[4, 3, 8], [-1, -2, 1]],
            [[3, 1, 6], [-2, -4, 2]],
        ]

        That is, the dimension is (6, 2, 3), where 6 is the number of functions, 2 is the number of variables,
        and 3 is the parameters for specifying a triangular number.

        It is worth noting that the parameters are set in the following order: [modal value, left boundary, right boundary]
        [4, 1, 7], where 4 is the modal value, 1 is the left boundary, and 7 is the right boundary

    The constraints for the optimization problem are also set.:

        1) 2 * x1 + 2 * x2 <= 8;

            This limitation means that the total number of hours
            that can be allocated to increase productivity (x1)
            and improve product quality/customer satisfaction/marketing efforts (x2)
            should not exceed 8 hours per day.

        2) -x1 + 3 * x2 <= 9;

            This constraint indicates that the number of hours
            can be spent on improving product quality/customer satisfaction/marketing efforts (x2)
            should not exceed 9 hours per day, provided
            that at least 1 hour is spent on improving productivity (x1).

        3) 2 * x1 - x2 <= 10;

            This constraint shows that the amount of time spent on minimizing equipment downtime (x1)
            and marketing efforts (x2) should not exceed 10 hours per day.

        4) x1 >= 0, x2 >= 0; The number of hours should not be negative
"""

# (The library is already installed in your project)
from fuzzyops.fuzzy_optimization import LinearOptimization, calc_total_functions,\
    get_interaction_matrix, check_LR_type, calc_total_functions
from fuzzyops.fuzzy_numbers import Domain

import numpy as np

coefs_domain = Domain((-20, 50, 1), name='coefs')
n = 5

# we set the fuzzy coefficients for the target functions to determine the interaction coefficients
f_num_params = np.array([
    [[4, 1, 7], [2, 0, 5]],
    [[2, 1, 3], [4, 2, 6]],
    [[3, -1, 4], [9, 5, 12]],
    [[4, 3, 8], [-1, -2, 1]],
    [[3, 1, 6], [-2, -4, 2]],
])

# we take modal values
C = f_num_params[:, :, 0]

C_f = []

for i in range(len(f_num_params)):
    lst = []
    for j in range(len(f_num_params[i])):
        coefs = [f_num_params[i][j].tolist()[1], f_num_params[i][j].tolist()[0], f_num_params[i][j].tolist()[2]]
        lst.append(coefs_domain.create_number('triangular', *coefs, name=f"c{i}{j}"))
    C_f.append(np.array(lst))

C_f = np.array(C_f)

# we check the correspondence of fuzzy numbers with coefficients that correspond to the LR type
assert check_LR_type(C_f)

# the matrix of coefficients of constraints
A = np.array([[2, 3],
              [-1, 3],
              [2, -1]])

b = np.array([18, 9, 10])

# Find the coefficients and the table of how the functions relate to each other (Cooperate, Conflict, or Are Independent)
alphas, interactions_list = get_interaction_matrix(f_num_params)
# We construct the coefficients for the variables in the generalized objective function using the algorithm
final_coefs = calc_total_functions(alphas, C, interactions_list, n)

C_new = np.array([[final_coefs[0], final_coefs[1]]])

# Solving the optomization problem
opt = LinearOptimization(A, b, C_new, "max")
# we get optimal values
_, v = opt.solve_cpu()
print(v)