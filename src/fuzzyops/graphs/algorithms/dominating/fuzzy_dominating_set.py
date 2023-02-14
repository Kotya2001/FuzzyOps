"""
get dominating set from graph, where have not 'any connection between nodes',
but connection stronger than given fuzzy number.

number should be given in same way as numbers in edges of graph.
"""

from ...fuzzgraph import FuzzyGraph


def fuzzy_dominating_set(graph, number_value):
    if not(type(graph) is FuzzyGraph):
        raise Exception('"graph" can be only FuzzGraph instance')

    number = graph._edge_number_class(number_value, **graph._edge_params)

    curr_nodes = set(graph.get_nodes_list())


    to_ret_set = set()

    while curr_nodes:
        curr_n = curr_nodes.pop()
        to_ret_set.add(curr_n)

        to_rm = graph.get_stronger_directly_connected(curr_n, number)
        for adj_node in to_rm:
            curr_nodes.discard(adj_node)

    return to_ret_set