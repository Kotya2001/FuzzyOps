"""
Task:

It is necessary to make a decision using the theory of fuzzy sets in the field of lending.
Suppose that a branch of a bank received applications from four enterprises for a loan.
The bank faces the task of choosing one enterprise, the best in terms of a complex of quality criteria.
In this task, the enterprises are alternatives (a1, a2, a3, a4).

To assess the creditworthiness of borrowers, some indicators from their accounting statements are used and the
following coefficients are calculated based on these data:

    1.Absolute Liquidity Ratio (F1);
    2.Intermediate coverage coefficient (F2);
    3.Total Coverage Ratio (F3);
    4.Financial Independence coefficient (F4);
    5.Product profitability ratio (F5);

The coefficients are calculated using formulas based on data from the financial statements.
Let's omit their calculations and present the final matrix of values:

        a1    a2    a3    a4    Normative values
    F1  0.154 0.102 0.084 0.14  0.1 - 0.25
    F2  1.297 0.71  0.59  0.57  0.5 - 1.0
    F3  2.78  2.27  1.86  1.27  1.0 - 2.5
    F4  0.75  0.72  0.71  0.68  0.6 - 0.8
    F5  0.28  0.115 0.15  0.12  The higher, the better


Next, it is necessary to create domain domains for the values of each coefficient (F1, F2, ..., F5),
then it is necessary to construct membership functions (fuzzy terms), that is, to set
the coefficients of the proximity functions (set by the expert).
Let's create one term and call it the "most preferred coefficient value" (F1, F2, ...)

After creating the fuzzy numbers, it is necessary to find the degrees of confidence for each value from the matrix
(For example, find the degree of confidence for 0.154 for the term "most preferred coefficient value" F1, etc.)

After that, in the new matrix of degrees of confidence, 
we find the minima by columns, and from the resulting vector (we get a vector of size 1x4 (4 enterprises))
we find argmax - this will be the resulting decision - which enterprise is most preferable to give a loan to.

"""

# (The library is already installed in your project)
from fuzzyops.fuzzy_numbers import Domain
import torch

# financial ratio matrix (5x4, 5 - number of ratios (F1, F2, ..., F5), 4 - number of enterprises)
coefs = torch.Tensor([[0.154, 0.102, 0.084, 0.140],
                      [1.297, 0.71, 0.59, 0.57],
                      [2.78, 2.27, 1.86, 1.27],
                      [0.75, 0.72, 0.71, 0.68],
                      [0.28, 0.115, 0.15, 0.12]])

# we create domains for each financial coefficient
f1 = Domain((0, 0.3, 0.05), name='f1')
f2 = Domain((0, 1.5, 0.05), name='f2')
f3 = Domain((0, 4.5, 0.05), name='f3')
f4 = Domain((0, 1, 0.05), name='f4')
f5 = Domain((0, 2, 0.05), name='f5')

# we create fuzzy numbers that determine the most preferred value according to the expert
f1.create_number('trapezoidal', 0.06, 0.19, 0.3, 0.3, name="most")
f2.create_number('trapezoidal', 0, 1, 1.5, 1.5, name="most")
f3.create_number('trapezoidal', 0, 2.5, 4.5, 4.5, name="most")
f4.create_number('trapezoidal', 0, 0.75, 1, 1, name="most")
f5.create_number('trapezoidal', 0, 0.3, 2, 2, name="most")

f_nums = [f1.most, f2.most, f3.most, f4.most, f5.most]
new_coefs = torch.zeros_like(coefs)

# we find the degrees of confidence for each value from the matrix with coefficient values for each enterprise
for i in range(coefs.size(0)):  # Iterating along the lines
    new_coefs[i] = torch.tensor([f_nums[i](value).item() for value in coefs[i]])

# we find the minimum values among the columns
min_values = torch.min(new_coefs, dim=0)[0]
# we find which company has the maximum value
print(torch.argmax(min_values))