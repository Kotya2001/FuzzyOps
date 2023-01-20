"""
finding the shortest path between two given nodes on graph,
if these nodes are connected.
"""

from ...fuzzgraph import FuzzyGraph

def shortest_path(graph, start_node, end_node):
    if not(type(graph) is FuzzyGraph):
        raise Exception('Can use only FuzzGraph')

    if not(graph.check_node(start_node) and graph.check_node(end_node)):
        raise Exception('No such nodes in graph')

    # initialise all needed and check outgoing lengths for start node
    common_nodes = graph.get_directly_connected(start_node)
    shortest_lengths = {val: [
        graph.get_edge_len(start_node, val),    # len of path
        [start_node, val]                       # path itself
    ]  for val in common_nodes}
    to_check = set(common_nodes)

    # traverse all connected nodes
    while len(to_check) != 0:
        curr_node = to_check.pop()
        common_nodes = graph.get_directly_connected(curr_node)
        for new_node in common_nodes:
            new_len = shortest_lengths[curr_node][0] + graph.get_edge_len(curr_node, new_node)
            old_len = shortest_lengths.get(new_node, None)
            if (old_len is None) or (old_len[0] > new_len):
                shortest_lengths[new_node] = [new_len, [*(shortest_lengths[curr_node][1]), new_node]]
                to_check.add(new_node)

    if not(end_node in shortest_lengths):
        raise Exception('start node and end node are not connected')

    return {
        'path': shortest_lengths[end_node][1],
        'len': shortest_lengths[end_node][0],
    }


