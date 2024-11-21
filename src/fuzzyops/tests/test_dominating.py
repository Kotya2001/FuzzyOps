import unittest
import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.graphs.algorithms.dominating import dominating_set, \
    fuzzy_dominating_set, is_dominating


class TestDominating(unittest.TestCase):
    """
    Тестирование алгоритмов нечеткого диминирования
    """
    def setUp(self):
        self.graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base', )
        for i in range(10):
            self.graph.add_node()

        self.graph.add_edge(0, 2, [3, 1, 1])
        self.graph.add_edge(0, 1, [2, 1, 1])
        self.graph.add_edge(1, 3, [5, 1, 1])
        self.graph.add_edge(1, 4, [1, 1, 1])
        self.graph.add_edge(2, 5, [6, 1, 1])
        self.graph.add_edge(2, 3, [4, 1, 1])
        self.graph.add_edge(3, 6, [3, 1, 1])
        self.graph.add_edge(4, 7, [4, 1, 1])
        self.graph.add_edge(5, 6, [4, 1, 1])
        self.graph.add_edge(5, 8, [2, 1, 1])
        self.graph.add_edge(6, 7, [5, 1, 1])
        self.graph.add_edge(7, 9, [3, 1, 1])
        self.graph.add_edge(8, 9, [3, 1, 1])

    def test_is_dominating(self):
        """
        Проверяет, является ли заданное множество узлов доминирующим в нечетком графе.

        Доминирующее множество - это подмножество узлов графа, такое что
        каждый узел графа либо принадлежит этому подмножеству, либо смежен с
        хотя бы одним узлом из этого подмножества.

        """
        dom1 = is_dominating(self.graph, {8, 9, 6, 0})
        dom2 = is_dominating(self.graph, {8, 9, 6, 0, 1})

        self.assertFalse(dom1)
        self.assertTrue(dom2)
        self.assertRaises(Exception, is_dominating, self.graph, {5, 9, 1, 12})

    def test_dominating_set(self):
        """
        Находит любое доминирующее множество в заданном нечетком графе.

        """
        fd = dominating_set(self.graph)
        res = is_dominating(self.graph, fd)

        self.assertTrue(res)

    def test_fuzzy_dominating_set(self):
        """
        Находит доминирующее множество в заданном нечетком графе,
        где соединение между узлами должно быть сильнее заданного нечеткого числа.

        """
        fd = fuzzy_dominating_set(self.graph, [3, 1, 1])
        res = is_dominating(self.graph, fd)

        self.assertTrue(res)
