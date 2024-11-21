


from fuzzyops.graphs.fuzzgraph import FuzzyGraph


def dominating_set(graph: FuzzyGraph) -> set:
    """
    Находит любое доминирующее множество в заданном нечетком графе.

    Доминирующее множество - это подмножество узлов графа, такое что
    каждый узел графа либо принадлежит этому подмножеству, либо смежен с
    хотя бы одним узлом из этого подмножества.

    Args:
        graph (FuzzyGraph): Экземпляр нечеткого графа.

    Returns:
        set: Множество индексов узлов, входящих в доминирующее множество.

    Raises:
        Exception: Исключение возникает, если переданный граф не является
        экземпляром класса `FuzzyGraph`.
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
