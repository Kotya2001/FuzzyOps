"""
Задача:
    Упарвление некоторой фирмой по производству какой-либо продукции (например мебель)
    Необходимо увеличить производстов, но при этом есть некоторые ограничения:
        ограничение на повышение производительности труда,
        ограничение на улучшение качетсва продукции,
        ограничение на время проведения маркетинговых усилий


    Допустим, мы определили 5 ключевых показателей эффективности, которые хотим повысить

    1.Производительность труда (f1) - увеличение количества выпускаемой продукции за час;
    2.Качество продукции (f2) - Уменьшение числа дефектов на единицу продукции;
    3.Удовлетворенность клиентов (f3) - Повышение уровня удовлетворенности клиентов благодаря улучшению качества;
    4.Эффективность использования оборудования (f4) - минимизация времени простоя оборудования;
    5.Маркетинговые усилия (f5) - Увеличение доли рынка засчет продвижения продукции.

    При этом определим искомые переменные (x1, x2):

        x1 - количество часов, которое выделяется на повышение производительности труда (f1),
        и соответственно мининимазию времени простоя оборудования (f4),
        x2 - может обозначать количество часов, которое выделяется (f2) на улучшение качества продукции,
        удовлетворенности клиентов клиентов (f3), и маркетинговых усислий (f5)

    То есть задачи принимает следующий вид:

        f1(x) = A11 * x1 + B12 * x2 -> max;
        f2(x) = A21 * x1 + B22 * x2 -> max;
        f3(x) = A31 * x1 + B32 * x2 -> max;
        f4(x) = A41 * x1 + B42 * x2 -> max;
        f5(x) = A51 * x1 + B52 * x2 -> max;

    При этом коэффициенты при x1 и x2 в каждом выржении являются нечеткими числами типа LR (теругольное и унимодальное),
    точнее коэффициентами для задания нечеткого числа такого типа

    Например, для данной задачи коэффициенты будут выглядеть так:
        [
            [[4, 1, 7], [2, 0, 5]],
            [[2, 1, 3], [4, 2, 6]],
            [[3, -1, 4], [9, 5, 12]],
            [[4, 3, 8], [-1, -2, 1]],
            [[3, 1, 6], [-2, -4, 2]],
        ]

        То есть размерность (6, 2, 3), где 6 - это число функций, 2 - чило переменных,
         3 - параметры для задания теругольного числа

         Стоит отметить, что параметры задаются в таком порядке [модальное значени, левая граница, правая граница]
         [4, 1, 7], 4 - модальное значение, 1 - левая граница, 7 -правая граница

    Также задаются ограничения для задачи оптимизации:

        1) 2 * x1 + 2 * x2 <= 8;

            Это ограничение говорит о том, что общее количество часов,
            можно выделить на повышение производительности труда (x1)
            и улучшение качества продукции/удовлетворенности клиентов/маркетинговых усилий (x2)
            не должно превышать 8 часов в день.

        2) -x1 + 3 * x2 <= 9;

            Это ограничение указывает на то, что количество часов,
            можно потратить на улучшение качества продукции/удовлетворенности клиентов/маркетинговых усилий (х2)
            не должно превышать 9 часов в день, при условии,
            что тратится минимум 1 час на повышение производительности труда (х1).

        3) 2 * x1 - x2 <= 10;

            Это ограничение показывает что сумма времени, которая тратится на минимизацию простое ообрудования (х1)
            и маркетинговых усилий (х2) не должна превышать 10 часов в день.

        4) x1 >= 0, x2 >= 0; Кол-во часов не должно быть отрицательным
"""

# Импорт необходимых классов для построения нечеткого логического вывода по алгоритму Мамдини
# (Библиотека уже установлена в ваш проект)
from fuzzyops.fuzzy_optimization import LinearOptimization, calc_total_functions,\
    get_interaction_matrix, check_LR_type, calc_total_functions
from fuzzyops.fuzzy_numbers import Domain

import numpy as np

coefs_domain = Domain((-20, 50, 1), name='coefs')
n = 5

# задачем нечеткие коэффициента при целевых функицях, для определения коэффициентов взаимодейтсвия
f_num_params = np.array([
    [[4, 1, 7], [2, 0, 5]],
    [[2, 1, 3], [4, 2, 6]],
    [[3, -1, 4], [9, 5, 12]],
    [[4, 3, 8], [-1, -2, 1]],
    [[3, 1, 6], [-2, -4, 2]],
])

# берем модальные значения
C = f_num_params[:, :, 0]

C_f = []

for i in range(len(f_num_params)):
    lst = []
    for j in range(len(f_num_params[i])):
        coefs = [f_num_params[i][j].tolist()[1], f_num_params[i][j].tolist()[0], f_num_params[i][j].tolist()[2]]
        lst.append(coefs_domain.create_number('triangular', *coefs, name=f"c{i}{j}"))
    C_f.append(np.array(lst))

C_f = np.array(C_f)

# проверяем соответствие нечетких чисел с нашмим коэффициентами, что соответствуют LR-типу
assert check_LR_type(C_f)

# матрица коэффициентов ограничений
A = np.array([[2, 3],
              [-1, 3],
              [2, -1]])

b = np.array([18, 9, 10])

# Находим коэффициенты и таблицу как функции соотносятся друг с другом (Кооперируют, Конфликтуют, Независят)
alphas, interactions_list = get_interaction_matrix(f_num_params)
# Строим коэффициенты при переменных у обобщенной целевой функции по алгоритму
final_coefs = calc_total_functions(alphas, C, interactions_list, n)

C_new = np.array([[final_coefs[0], final_coefs[1]]])

# Рещаем задачу оптмизации
opt = LinearOptimization(A, b, C_new, "max")
_, v = opt.solve_cpu()
print(v) # [6, 2]