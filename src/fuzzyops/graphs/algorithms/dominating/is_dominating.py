"""
checking if lower graph is dominating above higher
"""

from ...fuzzgraph import FuzzyGraph


def is_dominating(graph, nodes_set):
    if not(type(graph) is FuzzyGraph):
        raise Exception('"graph" can be only FuzzGraph instance')

    if not(type(nodes_set) is set):
        raise Exception('"nodes_set" can be only set instance')

    for node in nodes_set:
        if not(graph.check_node(node)):
            raise Exception('No such nodes in graph')

    adj_nodes = nodes_set.copy()

    for node in nodes_set:
        for adj_node in graph.get_directly_connected(node):
            adj_nodes.add(adj_node)

    return graph.check_nodes_full(list(adj_nodes))

