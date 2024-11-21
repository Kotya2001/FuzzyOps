


from fuzzyops.graphs.fuzzgraph import FuzzyGraph


def is_dominating(graph: FuzzyGraph, nodes_set: set[int]) -> bool:
    """
    Проверяет, является ли заданное множество узлов доминирующим в нечетком графе.

    Доминирующее множество - это подмножество узлов графа, такое что
    каждый узел графа либо принадлежит этому подмножеству, либо смежен с
    хотя бы одним узлом из этого подмножества.

    Args:
        graph (FuzzyGraph): Экземпляр нечеткого графа.
        nodes_set (set[int]): Множество индексов узлов, которые необходимо проверить
                               на доминирующесть.

    Returns:
        bool: Возвращает `True`, если `nodes_set` является доминирующим множеством
              в графе, иначе возвращает `False`.

    Raises:
        Exception: Исключение возникает, если переданный граф не является
        экземпляром класса `FuzzyGraph` или если `nodes_set` не является множеством.
        Также возникает исключение, если в `nodes_set` есть узлы, которых нет в графе.
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
