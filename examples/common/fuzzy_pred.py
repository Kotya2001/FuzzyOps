"""
Регрессия с нечеткими данными методом наименьших квадратов

Задача: Оценка теплопроводности материала с учетом треугольных нечетких данных

Описание задачи:
Необходимо оценить зависимость теплопроводности λ(T) от температуры T на основе экспериментальных
данных с нечеткими погрешностями, представленными треугольными функциями принадлежности

Входные переменные - нечеткие переменные измеренной температуры и соответствующие нечеткие переменные измеренной теплопроводности
Выходные - коэффициенты функции теплопроводности a и b, где a - угловой коэффициент, b - свободный член, и RMSE решения
"""

from fuzzyops.prediction import fit_fuzzy_linear_regression, convert_fuzzy_number_for_lreg
from fuzzyops.fuzzy_numbers import Domain

temp_domain = Domain((0, 111, 0.01), name='Temperature')
# Четкие числа записываем в виде треугольных нечетких чисел без хвостов
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
# Правая граница числа независимой переменной должна быть на 1 меньше правой границы доменной области для этой
# переменной
X_test = convert_fuzzy_number_for_lreg(temp_domain.create_number('triangular', 98, 105, 110))

Y_pred = (X_test * a) + b

print(Y_pred.to_fuzzy_number())