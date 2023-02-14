"""
get dominating set from graph (not shortest dominating, just any dominating)
"""

from ...fuzzgraph import FuzzyGraph


def dominating_set(graph):
    if not(type(graph) is FuzzyGraph):
        raise Exception('"graph" can be only FuzzGraph instance')

    curr_nodes = set(graph.get_nodes_list())


    to_ret_set = set()

    while curr_nodes:
        curr_n = curr_nodes.pop()
        to_ret_set.add(curr_n)

        to_rm = graph.get_directly_connected(curr_n)
        for adj_node in to_rm:
            curr_nodes.discard(adj_node)

    return to_ret_set


