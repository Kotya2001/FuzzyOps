# импорт класса нечеткого графа
from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.graphs.algorithms.transport import shortest_path
from fuzzyops.graphs.algorithms.factoring import mle_clusterization_factors
from fuzzyops.graphs.algorithms.dominating import dominating_set, \
    fuzzy_dominating_set, is_dominating
# создание нечеткого графа неориентированного графа (параметр edge_type = 'undirected')
# с node_number_math_type (типом операций вычисления для вершин - 'min'),
# с node_number_eq_type (типом операций сравнени для вершин - 'max')
# с edge_number_math_type (типом операций вычисления для ребер - 'mean'),
# с edge_number_eq_type (типом операций сравнени для ребер - 'base')
# граф строится на треугольных числах
graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base', )

# добавление вершин
for i in range(10):
    graph.add_node()

# добавление ребер с назначением им нечеткого числа
graph.add_edge(0, 2, [3, 1, 1])
graph.add_edge(0, 1, [2, 1, 1])
graph.add_edge(1, 3, [5, 1, 1])
graph.add_edge(1, 4, [1, 1, 1])
graph.add_edge(2, 5, [6, 1, 1])
graph.add_edge(2, 3, [4, 1, 1])
graph.add_edge(3, 6, [3, 1, 1])
graph.add_edge(4, 7, [4, 1, 1])
graph.add_edge(5, 6, [4, 1, 1])
graph.add_edge(5, 8, [2, 1, 1])
graph.add_edge(6, 7, [5, 1, 1])
graph.add_edge(7, 9, [3, 1, 1])
graph.add_edge(8, 9, [3, 1, 1])


# Поиск кратчайшего пути между вершинами
computedPath = shortest_path(graph, 0, 9)

print(computedPath)


clusters = mle_clusterization_factors(graph, 2)

print(clusters)

"""
Проверяет, является ли заданное множество узлов доминирующим в нечетком графе.

Доминирующее множество - это подмножество узлов графа, такое что
каждый узел графа либо принадлежит этому подмножеству, либо смежен с
хотя бы одним узлом из этого подмножества.
"""
dom1 = is_dominating(graph, {8, 9, 6, 0})
dom2 = is_dominating(graph, {8, 9, 6, 0, 1})
print(dom1, dom2)

# Находит любое доминирующее множество в заданном нечетком графе.
fd = dominating_set(graph)
res = is_dominating(graph, fd)
print(res)

"""
Находит доминирующее множество верщин в заданном нечетком графе,
где соединение между узлами должно быть сильнее заданного нечеткого числа.
"""
fd = fuzzy_dominating_set(graph, [3, 1, 1])
res = is_dominating(graph, fd)

print(fd, res)
