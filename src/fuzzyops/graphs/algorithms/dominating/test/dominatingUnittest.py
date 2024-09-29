import unittest

from .....graphs.fuzzgraph.graph import FuzzyGraph
from ..dominating_set import  dominating_set
from ..fuzzy_dominating_set import fuzzy_dominating_set
from ..is_dominating import is_dominating


class TestDominating(unittest.TestCase):

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

        dom1 = is_dominating(self.graph, {8, 9, 6, 0})
        dom2 = is_dominating(self.graph, {8, 9, 6, 0, 1})


        self.assertFalse(dom1)
        self.assertTrue(dom2)
        self.assertRaises(Exception, is_dominating, self.graph, {5, 9, 1, 12})


    def test_dominating_set(self):

        fd = dominating_set(self.graph)
        res = is_dominating(self.graph, fd)

        self.assertTrue(res)


    def test_fuzzy_dominating_set(self):

        fd = fuzzy_dominating_set(self.graph, [3,1,1])
        res = is_dominating(self.graph, fd)

        self.assertTrue(res)


