"""
Least squares regression with fuzzy data

Task: Assessment of thermal conductivity of a material taking into account triangular fuzzy data

Task description:
It is necessary to estimate the dependence of thermal conductivity λ(T) on temperature T based on experimental
data with fuzzy errors represented by triangular membership functions

The input variables are the fuzzy variables of the measured temperature and the corresponding fuzzy variables of the measured thermal conductivity
Output coefficients of the thermal conductivity function a and b, where a is the angular coefficient, b is the free term, and RMSE solutions

"""

from fuzzyops.prediction import fit_fuzzy_linear_regression, convert_fuzzy_number_for_lreg
from fuzzyops.fuzzy_numbers import Domain

temp_domain = Domain((0, 111, 0.01), name='Temperature')
# We write even numbers as triangular odd numbers without tails
temp_values = [
    temp_domain.create_number('triangular', 18, 20, 22),
    temp_domain.create_number('triangular', 38, 40, 42),
    temp_domain.create_number('triangular', 58, 60, 62),
    temp_domain.create_number('triangular', 78, 80, 82),
    temp_domain.create_number('triangular', 98, 100, 102)
]

tran_domain = Domain((1, 2, 0.01), name="Transcalency")
tran_values = [
    tran_domain.create_number('triangular', 1.2, 1.25, 1.3),
    tran_domain.create_number('triangular', 1.28, 1.35, 1.42),
    tran_domain.create_number('triangular', 1.35, 1.45, 1.55),
    tran_domain.create_number('triangular', 1.5, 1.62, 1.74),
    tran_domain.create_number('triangular', 1.65, 1.8, 1.95)
]

a, b, error = fit_fuzzy_linear_regression(temp_values, tran_values)
print(a, b, error)
# The right boundary of the number of the independent variable must be 1 less than the right boundary of the domain domain for this
# of the variable
X_test = convert_fuzzy_number_for_lreg(temp_domain.create_number('triangular', 98, 105, 110))

Y_pred = (X_test * a) + b

print(Y_pred.to_fuzzy_number())