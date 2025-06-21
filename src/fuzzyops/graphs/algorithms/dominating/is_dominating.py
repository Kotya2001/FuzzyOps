


from fuzzyops.graphs.fuzzgraph import FuzzyGraph


def is_dominating(graph: FuzzyGraph, nodes_set: set[int]) -> bool:
    """
    Checks whether a given set of nodes is dominant in a fuzzy graph
    
    A dominant set is a subset of graph nodes such that
    each node of the graph either belongs to this subset or is adjacent to
    at least one node from this subset

    Args:
        graph (FuzzyGraph): An instance of a fuzzy graph
        nodes_set (set[int]): There are many node indexes that need to be checked
            for dominance

    Returns:
        bool: Returns `True` if `nodes_set' is the dominant set
            in the graph, otherwise it returns `False'

    Raises:
        Exception: An exception occurs if the passed graph is not an
            instance of the `FuzzyGraph` class or if the `nodes_set` is not a set
            There is also an exception if there are nodes in the nodes_set that are not in the graph
    """

    if not (type(graph) is FuzzyGraph):
        raise Exception('"graph" can be only FuzzGraph instance')

    if not (type(nodes_set) is set):
        raise Exception('"nodes_set" can be only set instance')

    for node in nodes_set:
        if not (graph.check_node(node)):
            raise Exception('No such nodes in graph')

    adj_nodes = nodes_set.copy()

    for node in nodes_set:
        for adj_node in graph.get_directly_connected(node):
            adj_nodes.add(adj_node)

    return graph.check_nodes_full(list(adj_nodes))
