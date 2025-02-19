import unittest
from random import uniform
from time import perf_counter

import sys
from pathlib import Path
import os

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber, memberships

from fuzzyops.fuzzy_msa import fuzzy_pareto_solver, fuzzy_sum_solver, \
    fuzzy_pairwise_solver, fuzzy_hierarchy_solver

sys.setrecursionlimit(1500)


class TestFuzzyMSA(unittest.TestCase):
    """
    Тестирование классических алгоритмов многокритериального анализа с нечеткими переменными
    """

    def testFuzzyParetoPrecision(self):
        """
        Тестирование Нечеткой границы Паретто
        """
        d = Domain((0, 101), name='d', method='minimax')

        d.create_number('triangular', 1, 5, 11, name="x11")
        d.create_number('triangular', 3, 5, 7, name='x12')
        d.create_number('triangular', 0, 9, 13, name='x13')
        d.create_number('triangular', 4, 5, 7, name='x14')

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

        pareto = fuzzy_pareto_solver(solutions)
        print(pareto)

        assert pareto == [
            [d.x11, d.x21],
            [d.x13, d.x23]
        ]

    def testFuzzyParetoSpeed(self):
        """
        Тестирование Нечеткой границы Паретто на скорость выполнения
        """
        d = Domain((0, 101), name='d', method='minimax')

        solutions = []

        for i in range(100):
            sol = []
            for j in range(10):
                name = f"x_{i}_{j}"
                d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                sol.append(getattr(d, name))
            solutions.append(sol)

        print("Граница Парето")

        start = perf_counter()
        pareto = fuzzy_pareto_solver(solutions)
        end = perf_counter()

        print("time: ", end - start)

    def testFuzzySumPrecision(self):
        """
        Тестирование Нечеткой взвешенной суммы
        """
        d = Domain((0, 101), name='d', method='minimax')

        d.create_number('triangular', 1, 5, 11, name='w1')
        d.create_number('triangular', 3, 5, 7, name='w2')
        d.create_number('triangular', 0, 9, 12, name='w3')

        d.create_number('triangular', 4, 5, 7, name='a1')
        d.create_number('triangular', 3, 6, 13, name='a2')
        d.create_number('triangular', 2, 7, 10, name='a3')

        d.create_number('triangular', 5, 6, 7, name='b1')
        d.create_number('triangular', 1, 4, 11, name='b2')
        d.create_number('triangular', 3, 6, 14, name='b3')

        d.create_number('triangular', 4, 5, 7, name='c1')
        d.create_number('triangular', 3, 6, 13, name='c2')
        d.create_number('triangular', 1, 2, 8, name='c3')

        criteria_weights = [d.w1, d.w2, d.w3]

        alternatives_scores = [
            [d.a1, d.a2, d.a3],
            [d.b1, d.b2, d.b3],
            [d.c1, d.c2, d.c3],
        ]

        result = fuzzy_sum_solver(criteria_weights, alternatives_scores)

        assert str(result) == '[Fuzzy6.175824165344238, Fuzzy7.151782989501953, Fuzzy4.645833492279053]'

    def testFuzzySumSpeedCPU(self):
        """
        Тестирование Нечеткой взвешенной суммы на скорость на CPU
        """
        alts = 1000
        crit = 10
        d = Domain((0, 101), name='d', method='minimax')
        # d.to('cpu')

        criteria_weights = []
        for i in range(crit):
            name = f"w_{i}"
            d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
            criteria_weights.append(getattr(d, name))

        alternatives_scores = []
        for i in range(alts):
            sc = []
            for j in range(crit):
                name = f"x_{i}_{j}"
                d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                sc.append(getattr(d, name))
            alternatives_scores.append(sc)

        print("Взвешенные суммы")

        start = perf_counter()
        result = fuzzy_sum_solver(criteria_weights, alternatives_scores)
        end = perf_counter()

        print("time: ", end - start)

    def testFuzzySumSpeedGPU(self):
        """
        Тестирование Нечеткой взвешенной суммы на скорость на GPU
        """
        alts = 1000
        crit = 10
        d = Domain((0, 101), name='d', method='minimax')
        d.to('cuda')

        criteria_weights = []
        for i in range(crit):
            name = f"w_{i}"
            d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
            criteria_weights.append(getattr(d, name))

        alternatives_scores = []
        for i in range(alts):
            sc = []
            for j in range(crit):
                name = f"x_{i}_{j}"
                d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                sc.append(getattr(d, name))
            alternatives_scores.append(sc)

        print("Взвешенные суммы")

        start = perf_counter()
        result = fuzzy_sum_solver(criteria_weights, alternatives_scores)
        end = perf_counter()

        print("time: ", end - start)

    def testFuzzyPairwisePrecision(self):
        """
        Тестирование Нечетких парных сравнений
        """
        d = Domain((0, 101), name='d', method='minimax')

        alternatives = ["Альтернатива 1", "Альтернатива 2", "Альтернатива 3"]
        criteria = ["Критерий 1", "Критерий 2", "Критерий 3"]

        d.create_number('triangular', 1, 2, 3, name='a11')
        d.create_number('triangular', 1, 2, 3, name='a12')
        d.create_number('triangular', 1, 2, 3, name='a13')

        d.create_number('triangular', 1, 2, 3, name='b11')
        d.create_number('triangular', 1, 2, 3, name='b12')
        d.create_number('triangular', 1, 2, 3, name='b13')

        d.create_number('triangular', 1, 2, 3, name='c11')
        d.create_number('triangular', 1, 2, 3, name='c12')
        d.create_number('triangular', 1, 2, 3, name='c13')

        d.create_number('triangular', 1, 2, 3, name='a21')
        d.create_number('triangular', 1, 2, 3, name='a22')
        d.create_number('triangular', 1, 2, 3, name='a23')

        d.create_number('triangular', 1, 2, 3, name='b21')
        d.create_number('triangular', 1, 2, 3, name='b22')
        d.create_number('triangular', 1, 2, 3, name='b23')

        d.create_number('triangular', 1, 2, 3, name='c21')
        d.create_number('triangular', 1, 2, 3, name='c22')
        d.create_number('triangular', 1, 2, 3, name='c23')

        d.create_number('triangular', 1, 2, 3, name='a31')
        d.create_number('triangular', 1, 2, 3, name='a32')
        d.create_number('triangular', 1, 2, 3, name='a33')

        d.create_number('triangular', 1, 2, 3, name='b31')
        d.create_number('triangular', 1, 2, 3, name='b32')
        d.create_number('triangular', 1, 2, 3, name='b33')

        d.create_number('triangular', 1, 2, 3, name='c31')
        d.create_number('triangular', 1, 2, 3, name='c32')
        d.create_number('triangular', 1, 2, 3, name='c33')

        pairwise_matrices = [
            [
                [d.a11, d.a12, d.a13],
                [d.b11, d.b12, d.b13],
                [d.c11, d.c12, d.c13],
            ],
            [
                [d.a21, d.a22, d.a23],
                [d.b21, d.b22, d.b23],
                [d.c21, d.c22, d.c23],
            ],
            [
                [d.a31, d.a32, d.a33],
                [d.b31, d.b32, d.b33],
                [d.c31, d.c32, d.c33],
            ]
        ]

        # Вызов функции
        result = fuzzy_pairwise_solver(alternatives, criteria, pairwise_matrices)
        print(result)

        assert str(
            result) == "[('Альтернатива 1', Fuzzy90.99999237060547), ('Альтернатива 2', Fuzzy90.99999237060547), " \
                       "('Альтернатива 3', Fuzzy90.99999237060547)]"

    def testFuzzyPairwiseSpeed(self):
        """
        Тестирование Нечетких парных сравнений на скорость
        """
        d = Domain((0, 101), name='d', method='minimax')

        alts = 20
        crits = 5

        alternatives = [f"alt_{i}" for i in range(alts)]
        criteria = [f"crit_{i}" for i in range(crits)]

        pairwise_matrices = []
        for cr in range(crits):
            matrix = []
            for i in range(alts):
                row = []
                for j in range(alts):
                    name = f"x_{cr}_{i}_{j}"
                    d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                    row.append(getattr(d, name))
                matrix.append(row)
            pairwise_matrices.append(matrix)

        print("Попарные сравнения")

        start = perf_counter()
        result = fuzzy_pairwise_solver(alternatives, criteria, pairwise_matrices)
        end = perf_counter()

        print("time: ", end - start)

    def testFuzzyHierarchyPrecision(self):
        """
        Тестирование Нечеткой аналитической иерархии
        """
        d = Domain((0, 101), name='d', method='minimax')

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
        d.create_number('triangular', 2, 7, 11, name='cc13')

        d.create_number('triangular', 5, 6, 7, name='cc21')
        d.create_number('triangular', 2, 7, 11, name='cc22')
        d.create_number('triangular', 1, 5, 11, name='cc23')

        d.create_number('triangular', 3, 5, 7, name='cc31')
        d.create_number('triangular', 0, 9, 13, name='cc32')
        d.create_number('triangular', 1, 5, 11, name='cc33')

        cost_comparisons = [
            [d.cc11, d.cc12, d.cc13],
            [d.cc21, d.cc22, d.cc23],
            [d.cc31, d.cc32, d.cc33],
        ]

        d.create_number('triangular', 5, 6, 7, name='qc11')
        d.create_number('triangular', 2, 7, 11, name='qc12')
        d.create_number('triangular', 1, 5, 11, name='qc13')

        d.create_number('triangular', 3, 6, 13, name='qc21')
        d.create_number('triangular', 3, 6, 13, name='qc22')
        d.create_number('triangular', 2, 7, 11, name='qc23')

        d.create_number('triangular', 3, 5, 7, name='qc31')
        d.create_number('triangular', 0, 9, 13, name='qc32')
        d.create_number('triangular', 1, 5, 11, name='qc33')

        quality_comparisons = [
            [d.qc11, d.qc12, d.qc13],
            [d.qc21, d.qc22, d.qc23],
            [d.qc31, d.qc32, d.qc33],
        ]

        d.create_number('triangular', 1, 5, 11, name='rc11')
        d.create_number('triangular', 3, 5, 7, name='rc12')
        d.create_number('triangular', 0, 9, 13, name='rc13')

        d.create_number('triangular', 4, 5, 7, name='rc21')
        d.create_number('triangular', 3, 6, 13, name='rc22')
        d.create_number('triangular', 2, 7, 11, name='rc23')

        d.create_number('triangular', 5, 6, 7, name='rc31')
        d.create_number('triangular', 1, 4, 7, name='rc32')
        d.create_number('triangular', 2, 7, 11, name='rc33')

        reliability_comparisons = [
            [d.rc11, d.rc12, d.rc13],
            [d.rc21, d.rc22, d.rc23],
            [d.rc31, d.rc32, d.rc33],
        ]

        alternative_comparisons = [cost_comparisons, quality_comparisons, reliability_comparisons]

        global_priorities = fuzzy_hierarchy_solver(criteria_weights, alternative_comparisons)

        assert str(global_priorities) == '[Fuzzy70.59259033203125, Fuzzy72.17252349853516, Fuzzy69.31375885009766]'

    def testFuzzyHierarchySpeed(self):
        """
        Тестирование Нечеткой аналитической иерархии на скорость
        """
        d = Domain((0, 101), name='d', method='minimax')

        alts = 25
        crits = 5

        criteria_weights = []
        for i in range(crits):
            row = []
            for j in range(crits):
                name = f"x_{i}_{j}"
                d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                row.append(getattr(d, name))
            criteria_weights.append(row)

        alternative_comparisons = []
        for cr in range(crits):
            matrix = []
            for i in range(alts):
                row = []
                for j in range(alts):
                    name = f"x_{cr}_{i}_{j}"
                    d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name=name)
                    row.append(getattr(d, name))
                matrix.append(row)
            alternative_comparisons.append(matrix)

        print("Аналитическая иерархия")

        start = perf_counter()
        result = fuzzy_hierarchy_solver(criteria_weights, alternative_comparisons)
        end = perf_counter()

        print("time: ", end - start)
