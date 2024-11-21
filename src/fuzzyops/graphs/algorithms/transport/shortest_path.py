


from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from typing import Dict


def shortest_path(graph: FuzzyGraph, start_node: int, end_node: int) -> Dict:
    """
    Находит кратчайший путь между двумя заданными узлами в нечетком графе.

    Узлы считаются связанными, если существует путь между ними, и для этого пути
    определяется его длина. Если узлы не связаны, выбрасывается исключение.

    Args:
        graph (FuzzyGraph): Экземпляр нечеткого графа, в котором необходимо найти путь.
        start_node (int): Индекс начального узла.
        end_node (int): Индекс конечного узла.

    Returns:
        Dict: Словарь с двумя ключами:
            - 'path': Список узлов, представляющих кратчайший путь от `start_node` до `end_node`.
            - 'len': Длина кратчайшего пути.

    Raises:
        Exception: Исключение возникает, если:
            - Переданный граф не является экземпляром класса `FuzzyGraph`.
            - Начальный или конечный узел не существуют в графе.
            - Начальный и конечный узлы не соединены (т.е. не существует пути между ними).
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


