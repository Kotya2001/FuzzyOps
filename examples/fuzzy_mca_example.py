from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.fuzzy_msa import fuzzy_pareto_solver, fuzzy_sum_solver, fuzzy_pairwise_solver, fuzzy_hierarchy_solver

"""
Задача:
Выбор поставщика для закупки оборудования

Компания выбирает среди четырёх поставщиков оборудования, учитывая два критерия: цена (cost) – чем ниже, тем лучше и 
качество (quality) – чем выше, тем лучше.

Каждый поставщик предлагает разные условия (цена и качество), представленные в виде нечетких чисел, отражающих 
неопределенность в оценке.

Метод границы Парето позволяет найти набор оптимальных поставщиков, у которых невозможно улучшить один 
показатель (например, качество) без ухудшения другого (цены). Этот метод помогает логистам и закупщикам выбрать 
лучших поставщиков без явного взвешивания критериев.

Кроме этого, можно определить важность каждого параметра через нечеткий коэффициент веса. Тогда, имея эти веса,
можно определить рейтинг каждой из альтернатив, найдя оптимальную с использованием нечеткой взвешенной суммы. 
Метод полезен при выборе среди комплексных альтернатив с несколькими характеристиками.

"""

d = Domain((-100, 1000), name='d', method='minimax')

######################################################
# 1. Определение границы Парето
######################################################


# Критерии: Цена и качество
d.create_number('triangular', 1, 5, 11, name='cost1')
d.create_number('triangular', 3, 5, 7, name='cost2')
d.create_number('triangular', 0, 9, 13, name='cost3')
d.create_number('triangular', 4, 5, 7, name='cost4')

d.create_number('triangular', 3, 6, 13, name='quality1')
d.create_number('triangular', 2, 7, 11, name='quality2')
d.create_number('triangular', 5, 6, 7, name='quality3')
d.create_number('triangular', 1, 4, 7, name='quality4')

# Альтернативы: 4 поставщика
alternatives_scores = [
    [d.cost1, d.quality1],
    [d.cost2, d.quality2],
    [d.cost3, d.quality3],
    [d.cost4, d.quality4],
]

# Граница Парето
pareto_front = fuzzy_pareto_solver(alternatives_scores)
print("Нечеткая граница Парето:", pareto_front)

######################################################
# 2. Определение нечеткой взвешенной суммы
######################################################

d.create_number('triangular', 1, 5, 11, name='weight1')
d.create_number('triangular', 3, 5, 7, name='weight2')

criteria_weights = [d.weight1, d.weight2]

# Взвешенная сумма
weighted_sum = fuzzy_sum_solver(criteria_weights, alternatives_scores)
print("Нечеткая взвешенная сумма:", weighted_sum)

"""

Задача:
Выбор лучшего варианта оборудования среди нескольких моделей

ЛПР выбирает между тремя моделями промышленного оборудования для компании, учитывая стоимость (дешевле — лучше) и 
качество (лучше качество — лучше ). 

Все смартфоны сравниваются попарно по стоимости и качеству, используя нечеткие оценки 
(например, «немного лучше», «намного лучше» и т. д.). Метод попарных сравнений позволяет ранжировать варианты, 
выявляя лучшую, учитывая все критерии.

"""

######################################################
# 3. Определение нечетких попарных сравнений
######################################################


alternatives = ["Оборудование 1", "Оборудование 2", "Оборудование 3"]
criteria = ["Стоимость", "Качество"]

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
Задача: 
Выбор лучшего кандидата на должность менеджера

HR-отдел компании анализирует кандидатов на позицию менеджера, оценивая их по трём основным критериям стоимость найма 
(зарплатные ожидания, расходы на обучение), качество работы (опыт, навыки), надёжность (уровень ответственности, 
рекомендации). Каждый критерий разбивается является собирательным, например, «качество работы» учитывает и опыт, и 
лидерские качества.

Метод аналитической иерархии позволяет разделить сложное решение на уровни, сравнивать кандидатов по всем параметрам и 
выбрать лучшего. Используется в подборе персонала, стратегическом планировании и инвестиционных решениях.

"""

######################################################
# 4. Определение нечеткой иерархии
######################################################

d.create_number('triangular', 1, 5, 11, name='cw11')
d.create_number('triangular', 3, 5, 7, name='cw12')
d.create_number('triangular', 0, 9, 13, name='cw13')

d.create_number('triangular', 4, 5, 7, name='cw21')
d.create_number('triangular', 3, 6, 13, name='cw22')
d.create_number('triangular', 2, 7, 11, name='cw23')

d.create_number('triangular', 5, 6, 7, name='cw31')
d.create_number('triangular', 1, 4, 7, name='cw32')
d.create_number('triangular', 2, 7, 11, name='cw33')

criteria_weights = [
    [d.cw11, d.cw12, d.cw13],
    [d.cw21, d.cw22, d.cw23],
    [d.cw31, d.cw32, d.cw33],
]

d.create_number('triangular', 3, 6, 13, name='cc11')
d.create_number('triangular', 3, 6, 13, name='cc12')

d.create_number('triangular', 5, 6, 7, name='cc21')
d.create_number('triangular', 2, 7, 11, name='cc22')

cost_comparisons = [
    [d.cc11, d.cc12],
    [d.cc21, d.cc22],
]

d.create_number('triangular', 5, 6, 7, name='qc11')
d.create_number('triangular', 2, 7, 11, name='qc12')

d.create_number('triangular', 3, 6, 13, name='qc21')
d.create_number('triangular', 3, 6, 13, name='qc22')

quality_comparisons = [
    [d.qc11, d.qc12],
    [d.qc21, d.qc22],
]

d.create_number('triangular', 1, 5, 11, name='rc11')
d.create_number('triangular', 3, 5, 7, name='rc12')

d.create_number('triangular', 4, 5, 7, name='rc21')
d.create_number('triangular', 3, 6, 13, name='rc22')

reliability_comparisons = [
    [d.rc11, d.rc12],
    [d.rc21, d.rc22],
]

alternative_comparisons = [cost_comparisons, quality_comparisons, reliability_comparisons]

# Аналитическая иерархия
hierarchy_result = fuzzy_hierarchy_solver(criteria_weights, alternative_comparisons)
print("Нечеткая аналитическая иерархия:", hierarchy_result)
