from . import math
from . import fuzzify
from . import _fuzzynumber

"""
# example:

# create graph:
from fuzzyops.graphs.fuzzgraph.graph import FuzzyGraph

g = FuzzyGraph(
    edge_type='undirected',
    edge_number_type='triangle',
    edge_number_math_type='mean',
    edge_number_eq_type='max'
)

for i in range(10):
    g.add_node()

g.add_edge(0, 2, [3,1,1])
g.add_edge(0, 1, [2,1,1])
g.add_edge(1, 3, [5,1,1])
g.add_edge(1, 4, [1,1,1])
g.add_edge(2, 5, [6,1,1])
g.add_edge(2, 3, [4,1,1])
g.add_edge(3, 6, [3,1,1])
g.add_edge(4, 7, [4,1,1])
g.add_edge(5, 6, [4,1,1])
g.add_edge(5, 8, [2,1,1])
g.add_edge(6, 7, [5,1,1])
g.add_edge(7, 9, [3,1,1])
g.add_edge(8, 9, [3,1,1])


# the shortest part algorithm:
from fuzzyops.graphs.algorithms.transport import shortest_path
shp = shortest_path(g, 0, 9)
print(shp)
"""