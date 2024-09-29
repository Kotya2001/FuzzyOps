import unittest
from random import uniform

from ...fuzzy_numbers import Domain
from ...fuzzy_msa import fuzzy_pareto_solver, fuzzy_sum_solver, fuzzy_pairwise_solver, fuzzy_hierarchy_solver

class TestFuzzyMSA(unittest.TestCase):

    def testFuzzyPareto(self):
        d = Domain((0, 101), name='d', method='minimax')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='x11')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='x12')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='x13')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='x14')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='x21')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='x22')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='x23')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='x24')

        solutions = [
            [d.x11, d.x21],
            [d.x12, d.x22],
            [d.x13, d.x23],
            [d.x14, d.x24],
        ]

        pareto = fuzzy_pareto_solver(solutions)
        print("Граница Парето:")
        for solution in pareto:
            print(solution)


    def testFuzzySum(self):
        d = Domain((0, 101), name='d', method='minimax')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='w1')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='w2')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='w3')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a1')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a2')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a3')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b1')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b2')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b3')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c1')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c2')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c3')

        criteria_weights = [d.w1, d.w2, d.w3]

        alternatives_scores = [
            [d.a1, d.a2, d.a3],
            [d.b1, d.b2, d.b3],
            [d.c1, d.c2, d.c3],
        ]

        result = fuzzy_sum_solver(criteria_weights, alternatives_scores)

        for i in range(len(result)):
            print(f"Альтернатива {i+1}: Итоговая взвешенная оценка = {result[i]}")


    def testFuzzyPairwise(self):

        d = Domain((0, 101), name='d', method='minimax')

        alternatives = ["Альтернатива 1", "Альтернатива 2", "Альтернатива 3"]
        criteria = ["Критерий 1", "Критерий 2", "Критерий 3"]

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a11')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a12')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a13')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b11')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b12')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b13')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c11')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c12')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c13')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a21')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a22')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a23')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b21')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b22')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b23')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c21')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c22')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c23')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a31')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a32')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='a33')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b31')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b32')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='b33')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c31')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c32')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='c33')



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

        # Вывод результата
        for alt, score in result:
            print(f"{alt}: {score}")


    def testFuzzyHierarchy(self):
        d = Domain((0, 101), name='d', method='minimax')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw11')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw12')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw13')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw21')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw22')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw23')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw31')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw32')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cw33')

        criteria_weights = [
            [d.cw11, d.cw12, d.cw13],
            [d.cw21, d.cw22, d.cw23],
            [d.cw31, d.cw32, d.cw33],
        ]


        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc11')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc12')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc13')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc21')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc22')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc23')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc31')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc32')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='cc33')

        cost_comparisons = [
            [d.cc11, d.cc12, d.cc13],
            [d.cc21, d.cc22, d.cc23],
            [d.cc31, d.cc32, d.cc33],
        ]


        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc11')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc12')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc13')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc21')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc22')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc23')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc31')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc32')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='qc33')

        quality_comparisons = [
            [d.qc11, d.qc12, d.qc13],
            [d.qc21, d.qc22, d.qc23],
            [d.qc31, d.qc32, d.qc33],
        ]

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc11')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc12')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc13')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc21')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc22')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc23')

        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc31')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc32')
        d.create_number('triangular', uniform(-20, -5), uniform(-5, 5), uniform(5, 20), name='rc33')

        reliability_comparisons = [
            [d.rc11, d.rc12, d.rc13],
            [d.rc21, d.rc22, d.rc23],
            [d.rc31, d.rc32, d.rc33],
        ]

        alternative_comparisons = [cost_comparisons, quality_comparisons, reliability_comparisons]

        global_priorities = fuzzy_hierarchy_solver(criteria_weights, alternative_comparisons)

        for i, priority in enumerate(global_priorities):
            print(f"Альтернатива A{i + 1}: приоритет = {priority}")

