import unittest

from .....graphs.fuzzgraph.graph import FuzzyGraph
from ..mle_clusterization_factors import  mle_clusterization_factors


class TestFactoring(unittest.TestCase):

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


    def test_mle_clustering(self):

        clusters = mle_clusterization_factors(self.graph, 2)

        self.assertEqual(clusters, [0, 1, 1, 1, 0, 0, 0, 1, 1, 1])






