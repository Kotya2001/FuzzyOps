import unittest
import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.graphs.fuzzgraph.numbers import GraphTriangleFuzzyNumber
from fuzzyops.sequencing_assignment import FuzzySASolver


class TestSASolver(unittest.TestCase):
    """
    Тестирование алгоритма задачи о назначениях на нечетком графе
    """

    def testCaseFullGraph(self):
        """
        Тестирование полного графа (с равным числом исполнителей и задач)
        каждый исполнитель может быть назначени на каждую задачу
        """
        graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base',
        )

        solver = FuzzySASolver()
        solver.load_graph(graph)

        solver.load_tasks_workers(
            [
                'task_A',
                'task_B',
                'task_C'
            ],
            [
                'worker_X',
                'worker_Y',
                'worker_Z'
            ]
        )

        solver.load_task_worker_pair_value('task_A', 'worker_X', [6, 1, 2])
        solver.load_task_worker_pair_value('task_A', 'worker_Y', [3, 1, 1])
        solver.load_task_worker_pair_value('task_A', 'worker_Z', [2, 1, 2])
        solver.load_task_worker_pair_value('task_B', 'worker_X', [4, 2, 1])
        solver.load_task_worker_pair_value('task_B', 'worker_Y', [5, 1, 1])
        solver.load_task_worker_pair_value('task_B', 'worker_Z', [1, 1, 1])
        solver.load_task_worker_pair_value('task_C', 'worker_X', [2, 1, 1])
        solver.load_task_worker_pair_value('task_C', 'worker_Y', [3, 2, 1])
        solver.load_task_worker_pair_value('task_C', 'worker_Z', [2, 1, 2])

        result = solver.solve()

        expected = {
            'assignments': [
                ['worker_X', 'task_C'],
                ['worker_Y', 'task_A'],
                ['worker_Z', 'task_B']
            ],
            'cost': GraphTriangleFuzzyNumber([6, 1.0, 1.0])
        }

        self.assertEqual(result, expected)

    def testCaseSparseGraph(self):
        """
        Тестирование случая (число исполнителей равно числу работ)
        Каждый исполнитель может быть назначен какой-либо задаче
        """
        graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base',
        )

        solver = FuzzySASolver()
        solver.load_graph(graph)

        solver.load_tasks_workers(
            [
                'task_A',
                'task_B',
                'task_C'
            ],
            [
                'worker_X',
                'worker_Y',
                'worker_Z'
            ]
        )

        solver.load_task_worker_pair_value('task_A', 'worker_X', [6, 1, 2])
        solver.load_task_worker_pair_value('task_A', 'worker_Y', [3, 1, 1])
        solver.load_task_worker_pair_value('task_B', 'worker_Z', [3, 1, 2])
        solver.load_task_worker_pair_value('task_C', 'worker_X', [2, 1, 1])
        solver.load_task_worker_pair_value('task_C', 'worker_Y', [3, 2, 1])
        solver.load_task_worker_pair_value('task_C', 'worker_Z', [1, 1, 1])

        result = solver.solve()

        expected = {
            'assignments': [
                ['worker_X', 'task_C'],
                ['worker_Y', 'task_A'],
                ['worker_Z', 'task_B']
            ],
            'cost': GraphTriangleFuzzyNumber([8, 1.0, 1.5])
        }

        self.assertEqual(result, expected)

    def testCaseWorkersGTTasks(self):
        """
        Тестирование, где число работников может быть больше числа задач
        и каждый исполнитель может быть назначен на любую задачу
        """
        graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base',
        )

        solver = FuzzySASolver()
        solver.load_graph(graph)

        solver.load_tasks_workers(
            [
                'task_A',
                'task_B',
            ],
            [
                'worker_W',
                'worker_X',
                'worker_Y',
                'worker_Z',
            ]
        )

        solver.load_task_worker_pair_value('task_A', 'worker_W', [2, 1, 1])
        solver.load_task_worker_pair_value('task_A', 'worker_X', [3, 1, 1])
        solver.load_task_worker_pair_value('task_A', 'worker_Y', [5, 1, 2])
        solver.load_task_worker_pair_value('task_A', 'worker_Z', [1, 2, 1])
        solver.load_task_worker_pair_value('task_B', 'worker_W', [4, 2, 1])
        solver.load_task_worker_pair_value('task_B', 'worker_X', [3, 1, 1])
        solver.load_task_worker_pair_value('task_B', 'worker_Y', [2, 1, 2])
        solver.load_task_worker_pair_value('task_B', 'worker_Z', [3, 2, 1])

        result = solver.solve()

        expected = {
            'assignments': [
                ['worker_W', 'task_A'],
                ['worker_X', 'task_B'],
                ['worker_Y', 'no assignment'],
                ['worker_Z', 'no assignment']
            ],
            'cost': GraphTriangleFuzzyNumber([5, 1.0, 1.0])
        }

        self.assertEqual(result, expected)

    def testCaseTasksGTWorkers(self):
        """
        Тестирование, где число исполнителей меньше числа задач, и каждый работнико может быть
        назначен на любую задачу
        """
        graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base',
        )

        solver = FuzzySASolver()
        solver.load_graph(graph)

        solver.load_tasks_workers(
            [
                'task_A',
                'task_B',
                'task_C',
                'task_D'
            ],
            [
                'worker_Y',
                'worker_Z'
            ]
        )

        solver.load_task_worker_pair_value('task_A', 'worker_Y', [6, 1, 2])
        solver.load_task_worker_pair_value('task_A', 'worker_Z', [3, 2, 1])
        solver.load_task_worker_pair_value('task_B', 'worker_Y', [4, 1, 1])
        solver.load_task_worker_pair_value('task_B', 'worker_Z', [5, 1, 2])
        solver.load_task_worker_pair_value('task_C', 'worker_Y', [3, 2, 1])
        solver.load_task_worker_pair_value('task_C', 'worker_Z', [2, 2, 1])
        solver.load_task_worker_pair_value('task_D', 'worker_Y', [4, 1, 1])
        solver.load_task_worker_pair_value('task_D', 'worker_Z', [1, 2, 2])

        result = solver.solve()

        expected = {
            'assignments': [
                ['worker_Y', 'task_C'],
                ['worker_Z', 'task_D']
            ],
            'cost': GraphTriangleFuzzyNumber([4, 2.0, 1.5])
        }

        self.assertEqual(result, expected)

    def testCaseNoAssignments(self):
        """
        Тестирование, где работники не назначаются на задачи (не создается двудольный граф)
        """
        graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base',
        )

        solver = FuzzySASolver()
        solver.load_graph(graph)

        solver.load_tasks_workers(
            [
                'task_A',
                'task_B',
                'task_C'
            ],
            [
                'worker_X',
                'worker_Y',
                'worker_Z'
            ]
        )

        result = solver.solve()

        expected = {
            'assignments': [
                ['worker_X', 'no assignment'],
                ['worker_Y', 'no assignment'],
                ['worker_Z', 'no assignment']
            ],
            'cost': None
        }

        self.assertEqual(result, expected)
