import unittest
import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.graphs.algorithms.factoring import mle_clusterization_factors


class TestFactoring(unittest.TestCase):
    """
    Testing of fuzzy factor models
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

    def test_mle_clustering(self):
        """
        A test of clustering nodes in a fuzzy graph using the MLE method

        """
        clusters = mle_clusterization_factors(self.graph, 2)

        self.assertEqual(clusters, [0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
