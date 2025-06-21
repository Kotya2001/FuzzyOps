


from fuzzyops.graphs.fuzzgraph import FuzzyGraph


def dominating_set(graph: FuzzyGraph) -> set:
    """
    Finds any dominant set in a given fuzzy graph
    
    A dominant set is a subset of graph nodes such that
    each node of the graph either belongs to this subset or is adjacent to
    at least one node from this subset

    Args:
        graph (FuzzyGraph): An instance of the fuzzy graph class

    Returns:
        set: The set of indexes of nodes included in the dominant set

    Raises:
        Exception: An exception occurs if the passed graph is not an
        instance of the `FuzzyGraph' class
    """

    if not (type(graph) is FuzzyGraph):
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
