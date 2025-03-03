from src.fuzzyops.graphs.fuzzgraph.graph import FuzzyGraph
from src.fuzzyops.graphs.algorithms.dominating import fuzzy_dominating_set
from src.fuzzyops.graphs.algorithms.factoring import mle_clusterization_factors
from src.fuzzyops.graphs.algorithms.transport import shortest_path



######################################################
# 1. Определение доминирующего подмножества
######################################################


graph_zones = FuzzyGraph(
    node_number_math_type='min',
    node_number_eq_type='max',
    edge_number_math_type='mean',
    edge_number_eq_type='base', )
for i in range(10):
    graph_zones.add_node()

graph_zones.add_edge(0, 2, [3, 1, 2])
graph_zones.add_edge(0, 1, [2, 1, 1])
graph_zones.add_edge(1, 3, [5, 2, 1])
graph_zones.add_edge(1, 4, [1, 1, 1])
graph_zones.add_edge(2, 5, [6, 1, 1])
graph_zones.add_edge(2, 3, [4, 2, 1])
graph_zones.add_edge(3, 6, [3, 1, 1])
graph_zones.add_edge(4, 7, [4, 1, 1])
graph_zones.add_edge(5, 6, [4, 1, 2])
graph_zones.add_edge(5, 8, [2, 1, 1])
graph_zones.add_edge(6, 7, [5, 2, 1])
graph_zones.add_edge(7, 9, [3, 1, 1])
graph_zones.add_edge(8, 9, [3, 2, 1])


fd = fuzzy_dominating_set(graph_zones, [3, 1, 1])

print(f"Доминирующее подмножество: {fd}")


######################################################
# 2. Сегментация клиентов
######################################################


graph_customers = FuzzyGraph(
    node_number_math_type='min',
    node_number_eq_type='max',
    edge_number_math_type='mean',
    edge_number_eq_type='base', )
for i in range(10):
    graph_customers.add_node()

graph_customers.add_edge(0, 2, [3, 1, 1])
graph_customers.add_edge(0, 1, [2, 2, 1])
graph_customers.add_edge(1, 3, [3, 1, 2])
graph_customers.add_edge(1, 4, [1, 2, 1])
graph_customers.add_edge(2, 5, [5, 1, 1])
graph_customers.add_edge(2, 3, [4, 2, 1])
graph_customers.add_edge(3, 6, [4, 1, 1])
graph_customers.add_edge(4, 7, [4, 1, 2])
graph_customers.add_edge(5, 6, [2, 1, 1])
graph_customers.add_edge(5, 8, [3, 1, 2])
graph_customers.add_edge(6, 7, [5, 2, 1])
graph_customers.add_edge(7, 9, [1, 1, 1])
graph_customers.add_edge(8, 9, [3, 2, 1])


clusters = mle_clusterization_factors(graph_customers, 2)

print(f"Разбиение на кластера: {clusters}")



######################################################
# 3. Определение оптимального пути
######################################################


transport_graph = FuzzyGraph(
    node_number_math_type='min',
    node_number_eq_type='max',
    edge_number_math_type='mean',
    edge_number_eq_type='base', )
for i in range(10):
    transport_graph.add_node()

transport_graph.add_edge(0, 2, [1, 1, 2])
transport_graph.add_edge(0, 1, [2, 1, 1])
transport_graph.add_edge(1, 3, [3, 2, 1])
transport_graph.add_edge(1, 4, [1, 1, 1])
transport_graph.add_edge(2, 5, [6, 1, 1])
transport_graph.add_edge(2, 3, [5, 2, 1])
transport_graph.add_edge(3, 6, [4, 1, 2])
transport_graph.add_edge(4, 7, [4, 1, 1])
transport_graph.add_edge(5, 6, [2, 1, 1])
transport_graph.add_edge(5, 8, [2, 1, 2])
transport_graph.add_edge(6, 7, [3, 1, 1])
transport_graph.add_edge(7, 9, [4, 2, 1])
transport_graph.add_edge(8, 9, [3, 1, 1])

path = shortest_path(transport_graph, 0, 9)

print(f"Кратчайший путь: {path}")