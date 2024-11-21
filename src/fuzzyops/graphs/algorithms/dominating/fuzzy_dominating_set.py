


from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.graphs.fuzzgraph.numbers import GraphTriangleFuzzyNumber


def fuzzy_dominating_set(graph: FuzzyGraph, number_value: GraphTriangleFuzzyNumber) -> set:
    """
    Находит доминирующее множество в заданном нечетком графе,
    где соединение между узлами должно быть сильнее заданного нечеткого числа.

    Доминирующее множество - это подмножество узлов графа, такое что
    каждый узел графа либо принадлежит этому подмножеству, либо
    смежен с хотя бы одним узлом из этого подмножества.
    Однако, в отличие от обычного доминирующего множества, здесь
    учитываются только связи, которые сильнее заданного нечеткого числа.

    Args:
        graph (FuzzyGraph): Экземпляр нечеткого графа.
        number_value (GraphTriangleFuzzyNumber): Нечеткое число, задающее минимальную силу
                                                  соединений для включения узлов
                                                  в доминирующее множество.

    Returns:
        set: Множество индексов узлов, входящих в доминирующее множество.

    Raises:
        Exception: Исключение возникает, если переданный граф не является
        экземпляром класса `FuzzyGraph`.
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
