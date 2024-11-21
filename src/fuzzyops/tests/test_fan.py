import unittest
import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber
from fuzzyops.fan import Graph, calc_final_scores


class TestFAN(unittest.TestCase):
    """
    Тестирование нечетких аналитических сетей
    """
    def testSimpleGraph(self):
        """
        Тестирование простого аналитического графа
        """
        graph = Graph()
        graph.add_edge('A', 'B', 0.8)
        graph.add_edge('B', 'C', 0.9)
        graph.add_edge('A', 'D', 0.7)
        graph.add_edge('D', 'C', 0.85)

        most_feasible_path = graph.find_most_feasible_path('A', 'C')
        if most_feasible_path:
            print(f"Most feasible path from A to C: {most_feasible_path}")
            print(f"Feasibility: {graph.calculate_path_fuzziness(most_feasible_path)}")
        else:
            print("No path found.")

        best_alternative, max_feasibility = graph.macro_algorithm_for_best_alternative()
        if best_alternative:
            print(f"Best alternative: {best_alternative}")
            print(f"Max Feasibility: {max_feasibility}")
        else:
            print("No alternatives found.")

        assert most_feasible_path == ['A', 'B', 'C']
        assert best_alternative == ['B', 'C']

    def testComplexGraph(self):
        """
        Тестирование нечеткой аналитической сети с расчетом оценок на основе нечетких чисел
        """
        score_domain = Domain((0, 1, 0.01), name='scores')

        score_domain.create_number('triangular', 0.4, 0.7, 0.9, name='complex_A')
        score_domain.create_number('triangular', 0.4, 0.76, 1, name='sources_A')
        A_score = calc_final_scores([score_domain.complex_A, score_domain.sources_A])

        score_domain.create_number('triangular', 0.5, 0.7, 0.95, name='complex_A2')
        score_domain.create_number('triangular', 0.4, 0.85, 1, name='sources_A2')
        A2_score = calc_final_scores([score_domain.complex_A2, score_domain.sources_A2])

        score_domain.create_number('triangular', 0.5, 0.77, 1, name='complex_B')
        score_domain.create_number('triangular', 0.44, 0.89, 1, name='sources_B')
        B_score = calc_final_scores([score_domain.complex_B, score_domain.sources_B])

        score_domain.create_number('triangular', 0.3, 0.6, 0.91, name='complex_C')
        score_domain.create_number('triangular', 0.4, 0.7, 1, name='sources_C')
        C_score = calc_final_scores([score_domain.complex_C, score_domain.sources_C])

        score_domain.create_number('triangular', 0.4, 0.61, 0.81, name='complex_D')
        score_domain.create_number('triangular', 0.43, 0.7, 1, name='sources_D')
        D_score = calc_final_scores([score_domain.complex_D, score_domain.sources_D])

        score_domain.create_number('triangular', 0.5, 0.6, 0.91, name='complex_E')
        score_domain.create_number('triangular', 0.5, 0.8, 1, name='sources_E')
        E_score = calc_final_scores([score_domain.complex_E, score_domain.sources_E])

        graph = Graph()
        graph.add_edge('Start', 'A', A_score)
        graph.add_edge('Start', "A2", A2_score)
        graph.add_edge('A', 'B', max(A_score, B_score))
        graph.add_edge("A2", 'B', max(A2_score, B_score))
        graph.add_edge('B', 'C', max(C_score, B_score))
        graph.add_edge('C', 'D', max(C_score, D_score))
        graph.add_edge('C', 'E', max(C_score, E_score))
        graph.add_edge('D', 'End', D_score)
        graph.add_edge('E', 'End', E_score)

        most_feasible_path = graph.find_most_feasible_path('Start', 'End')
        if most_feasible_path:
            print(f"Most feasible path: {most_feasible_path}")
            print(f"Feasibility: {graph.calculate_path_fuzziness(most_feasible_path)}")
        else:
            print("No path found.")

        assert most_feasible_path == ['Start', 'A2', 'B', 'C', 'E', 'End']
