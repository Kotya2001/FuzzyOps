"""
Задача:
Расположение сотовых вышек по районам

Есть набор точек, куда можно установить сотовые вышки, а так же показатели, какой силы будет сигнал в близлежащие
районы с каждой вышки. Уже есть некоторый набор вышек, необходимо определить является ли этот набор вышек достаточным,
определить минимально достаточный набор районов, установив в которые вышки у каждого района будет доступна связь.

Предлагается решить задачу с помощью нечеткого графа.

Строим нечеткий граф, показывающий близость районов между собой. Близость показывает силу сигнала, который будет в
соседнем районе, если в текущем поставить вышку.

1. Имеется уже существующий набор вышек. Проверяется, является ли данный набор вышек достаточным для покрытия всей
рассматриваемой территории.

2. Ищется минимальный подграф районов, в которых необходимо установить вышки для полного
покрытия района

3. Ищется минимальный подграф, только с указанием минимальной силы сигнала, которая должна быть в каждом районе

"""

from fuzzyops.graphs.fuzzgraph.graph import FuzzyGraph
from fuzzyops.graphs.algorithms.dominating import dominating_set, fuzzy_dominating_set, is_dominating
from fuzzyops.graphs.algorithms.factoring import mle_clusterization_factors
from fuzzyops.graphs.algorithms.transport import shortest_path

######################################################
# 1. Проверка доминирующего подмножества
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

check_set = {0, 3, 5, 9}
isDom = is_dominating(graph_zones, check_set)

print(f"Является ли подмножество {check_set} доминирующим: {'да' if isDom else 'нет'}")

######################################################
# 2. Определение доминирующего подмножества
######################################################


ds = dominating_set(graph_zones)

print(f"Доминирующее подмножество: {ds}")

######################################################
# 3. Определение нечеткого доминирующего подмножества
######################################################


fd = fuzzy_dominating_set(graph_zones, [3, 1, 1])

print(f"Нечеткое доминирующее подмножество: {fd}")

"""
Задача:
сегментация клиентов по группам

Между некоторыми покупателями есть данные о схожести их поведения, насколько близко или далеко они находятся друг от 
друга. Для проведения А/Б тестов необходимо разбить пользователей на два кластера по их схожести. 

Строим нечеткий граф, и указываем схожесть между клиентами. Ноды графа - клиенты, связи графа - их схожесть. 
После этого применяем метод mle_clusterization_factors, указывая необходимое количество кластеров.

"""

######################################################
# 4. Сегментация клиентов
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

"""
Задача:
Решение логистической задачи

Необходимо доставить груз из точки А в точку Б за минимальное время, при этом учитывая возможность пробок или других
факторов, влияющих на время преодоления пути. 

Строим нечеткий граф, который отображает карту дорог, на связях указывая нечеткое время преодоления определенных
участков дорог. Нечеткость означает вероятные пробки, светофоры, ремонтные работы или другие факторы, которые
могут влиять на время проезда.

После этого применяем функцию shortest_path, указывая начальную точку и конечную точку. После применения функции 
выводится список нод, по которым следует пройти, чтобы добраться из точки А в точку Б за минимальное время.

"""

######################################################
# 5. Определение оптимального пути
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
