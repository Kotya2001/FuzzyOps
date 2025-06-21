


from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from typing import Dict


def shortest_path(graph: FuzzyGraph, start_node: int, end_node: int) -> Dict:
    """
    Finds the shortest path between two given nodes in a fuzzy graph
    
    Nodes are considered connected if there is a path between them, and
    its length is determined for this path. If the nodes are not connected, an exception is thrown

    Args:
        graph (FuzzyGraph): An instance of a fuzzy graph in which a path must be found
        start_node (int): The index of the initial node
        end_node (int): The index of the end node

    Returns:
        Dict: A dictionary with two keys:
            - 'path': List of nodes representing the shortest path from `start_node` to `end_node`
            - 'len': The length of the shortest path

    Raises:
        Exception: An exception occurs if:
            - The passed graph is not an instance of the `FuzzyGraph` class
            - The start or end node does not exist in the graph
            - The start and end nodes are not connected (i.e. there is no path between them)
    """

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


