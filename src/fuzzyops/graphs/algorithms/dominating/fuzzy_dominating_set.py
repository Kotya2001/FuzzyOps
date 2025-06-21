


from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.graphs.fuzzgraph.numbers import GraphTriangleFuzzyNumber


def fuzzy_dominating_set(graph: FuzzyGraph, number_value: GraphTriangleFuzzyNumber) -> set:
    """
    Finds the dominant set in a given fuzzy graph,
    where the connection between the nodes should be stronger than a given fuzzy number
    
    A dominant set is a subset of graph nodes such that
    each node of the graph either belongs to this subset or
    is adjacent to at least one node from this subset
    However, unlike the usual dominant set,
    only connections that are stronger than a given fuzzy number are taken into account here

    Args:
        graph (FuzzyGraph): An instance of the fuzzy graph class
        number_value (GraphTriangleFuzzyNumber): A fuzzy number that specifies the minimum strength
            of connections to include nodes in the dominant set

    Returns:
        set: The set of indexes of nodes included in the dominant set

    Raises:
        Exception: An exception occurs if the passed graph is not an
        instance of the `FuzzyGraph' class
    """

    if not (type(graph) is FuzzyGraph):
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
