from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber, memberships

from fuzzyops.fuzzy_msa import fuzzy_pareto_solver, fuzzy_sum_solver, \
    fuzzy_pairwise_solver, fuzzy_hierarchy_solver

"""
Нечеткая граница Парето
"""

# Создание домена для нечетких чисел
d = Domain((0, 101), name='d', method='minimax')

# создание нечетких чисел для критерия 1
d.create_number('triangular', 1, 5, 11, name="x11")
d.create_number('triangular', 3, 5, 7, name='x12')
d.create_number('triangular', 0, 9, 13, name='x13')
d.create_number('triangular', 4, 5, 7, name='x14')

# создание нечетких чисел для критерия 2
d.create_number('triangular', 3, 6, 13, name='x21')
d.create_number('triangular', 2, 7, 11, name='x22')
d.create_number('triangular', 5, 6, 7, name='x23')
d.create_number('triangular', 1, 4, 7, name='x24')

solutions = [
    [d.x11, d.x21],
    [d.x12, d.x22],
    [d.x13, d.x23],
    [d.x14, d.x24],
]

# Определение нечеткой границы Парето
pareto = fuzzy_pareto_solver(solutions)
print(pareto)


"""
Нечеткая взвешенная сумма
"""

d.create_number('triangular', 1, 5, 11, name='weight1')
d.create_number('triangular', 3, 5, 7, name='weight2')

criteria_weights = [d.weight1, d.weight2]

# Взвешенная сумма
weighted_sum = fuzzy_sum_solver(criteria_weights, solutions)
print("Нечеткая взвешенная сумма:", weighted_sum)


"""
Нечеткие попарные сравнения
"""

# Названия альтернатив и критериев
alternatives = ["Продукт 1", "Продукт 2", "Продукт 3"]
criteria = ["Стоимость", "Качество"]

# Данные для матрицы попарных сравнений: для каждой альтернативы и каждого критерия мы строим
# матрицу попарных сравнений, в которой сравниваются альтернативы по степени предпочтения по заданному критерию.
d.create_number('triangular', 1, 2, 3, name='Cost11')
d.create_number('triangular', 1, 1, 3, name='Cost12')
d.create_number('triangular', 1, 2, 5, name='Cost13')

d.create_number('triangular', 1, 2, 4, name='Cost21')
d.create_number('triangular', 1, 2, 5, name='Cost22')
d.create_number('triangular', 2, 3, 3, name='Cost23')

d.create_number('triangular', 1, 2, 3, name='Cost31')
d.create_number('triangular', 2, 2, 2, name='Cost32')
d.create_number('triangular', 1, 2, 5, name='Cost33')

d.create_number('triangular', 1, 2, 5, name='Quality11')
d.create_number('triangular', 2, 3, 4, name='Quality12')
d.create_number('triangular', 1, 2, 3, name='Quality13')

d.create_number('triangular', 1, 2, 5, name='Quality21')
d.create_number('triangular', 1, 3, 4, name='Quality22')
d.create_number('triangular', 2, 2, 3, name='Quality23')

d.create_number('triangular', 1, 3, 4, name='Quality31')
d.create_number('triangular', 2, 3, 3, name='Quality32')
d.create_number('triangular', 1, 2, 4, name='Quality33')

# матрица попарных оценок всех продуктов по 2 критериям
pairwise_matrices = [
    [
        [d.Cost11, d.Cost12, d.Cost13],
        [d.Cost21, d.Cost22, d.Cost23],
        [d.Cost31, d.Cost32, d.Cost33],
    ],
    [
        [d.Quality11, d.Quality12, d.Quality13],
        [d.Quality21, d.Quality22, d.Quality23],
        [d.Quality31, d.Quality32, d.Quality33],
    ],
]
# Попарные сравнения
pairwise_result = fuzzy_pairwise_solver(alternatives, criteria, pairwise_matrices)
print("Нечеткие попарные сравнения:", pairwise_result)


"""
Нечеткая аналитическая иерархия
"""

d.create_number('triangular', 1, 5, 11, name='cw11')
d.create_number('triangular', 3, 5, 7, name='cw12')
d.create_number('triangular', 0, 9, 13, name='cw13')

d.create_number('triangular', 4, 5, 7, name='cw21')
d.create_number('triangular', 3, 6, 13, name='cw22')
d.create_number('triangular', 2, 7, 11, name='cw23')

d.create_number('triangular', 5, 6, 7, name='cw31')
d.create_number('triangular', 1, 4, 7, name='cw32')
d.create_number('triangular', 2, 7, 11, name='cw33')

# Веса по каждому критерию
criteria_weights = [
    [d.cw11, d.cw12, d.cw13],
    [d.cw21, d.cw22, d.cw23],
    [d.cw31, d.cw32, d.cw33],
]

d.create_number('triangular', 3, 6, 13, name='cc11')
d.create_number('triangular', 3, 6, 13, name='cc12')

d.create_number('triangular', 5, 6, 7, name='cc21')
d.create_number('triangular', 2, 7, 11, name='cc22')

# Оценки критериев (критерий стоимости) по каждой альтернативе
cost_comparisons = [
    [d.cc11, d.cc12],
    [d.cc21, d.cc22],
]

d.create_number('triangular', 5, 6, 7, name='qc11')
d.create_number('triangular', 2, 7, 11, name='qc12')

d.create_number('triangular', 3, 6, 13, name='qc21')
d.create_number('triangular', 3, 6, 13, name='qc22')

# Оценки критериев (критерий качества) по каждой альтернативе
quality_comparisons = [
    [d.qc11, d.qc12],
    [d.qc21, d.qc22],
]

d.create_number('triangular', 1, 5, 11, name='rc11')
d.create_number('triangular', 3, 5, 7, name='rc12')

d.create_number('triangular', 4, 5, 7, name='rc21')
d.create_number('triangular', 3, 6, 13, name='rc22')

# Оценки критериев (критерий надежности) по каждой альтернативе
reliability_comparisons = [
    [d.rc11, d.rc12],
    [d.rc21, d.rc22],
]

alternative_comparisons = [cost_comparisons, quality_comparisons, reliability_comparisons]

# Аналитическая иерархия
hierarchy_result = fuzzy_hierarchy_solver(criteria_weights, alternative_comparisons)
print("Нечеткая аналитическая иерархия:", hierarchy_result)


