# importing a fuzzy graph class
from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.graphs.algorithms.transport import shortest_path
from fuzzyops.graphs.algorithms.factoring import mle_clusterization_factors
from fuzzyops.graphs.algorithms.dominating import dominating_set, \
    fuzzy_dominating_set, is_dominating
# creating a fuzzy graph of an undirected graph (edge_type parameter = 'undirected')
# with node_number_math_type (type of calculation operations for vertices - 'min'),
# with node_number_eq_type (type of comparison operations for vertices - 'max')
# with edge_number_math_type (type of calculation operations for edges - 'mean'),
# with edge_number_eq_type (type of comparison operations for edges - 'base')
# the graph is built on triangular numbers
graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base', )

# adding vertices
for i in range(10):
    graph.add_node()

# adding edges with a fuzzy number assigned to them
graph.add_edge(0, 2, [3, 1, 1])
graph.add_edge(0, 1, [2, 1, 1])
graph.add_edge(1, 3, [5, 1, 1])
graph.add_edge(1, 4, [1, 1, 1])
graph.add_edge(2, 5, [6, 1, 1])
graph.add_edge(2, 3, [4, 1, 1])
graph.add_edge(3, 6, [3, 1, 1])
graph.add_edge(4, 7, [4, 1, 1])
graph.add_edge(5, 6, [4, 1, 1])
graph.add_edge(5, 8, [2, 1, 1])
graph.add_edge(6, 7, [5, 1, 1])
graph.add_edge(7, 9, [3, 1, 1])
graph.add_edge(8, 9, [3, 1, 1])


# Finding the shortest path between vertices
computedPath = shortest_path(graph, 0, 9)

print(computedPath)


clusters = mle_clusterization_factors(graph, 2)

print(clusters)

"""
Checks whether a given set of nodes is dominant in a fuzzy graph.

A dominant set is a subset of the nodes in the graph such that
every node in the graph either belongs to this subset or is adjacent to
at least one node in this subset.
"""
dom1 = is_dominating(graph, {8, 9, 6, 0})
dom2 = is_dominating(graph, {8, 9, 6, 0, 1})
print(dom1, dom2)

# Finds any dominant set in a given fuzzy graph.
fd = dominating_set(graph)
res = is_dominating(graph, fd)
print(res)

"""
Finds a dominating set of vertices in a given fuzzy graph,
where the connection between nodes must be stronger than a given fuzzy number.
"""
fd = fuzzy_dominating_set(graph, [3, 1, 1])
res = is_dominating(graph, fd)

print(fd, res)
