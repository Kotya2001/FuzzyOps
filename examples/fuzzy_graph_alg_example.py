"""
Task:
Location of cell towers by area

There is a set of points where cell towers can be installed, as well as indicators of how strong the signal will be to the surrounding
areas from each tower. There is already a certain set of towers, it is necessary to determine whether this set of towers is sufficient,
to determine the minimum sufficient set of areas, by installing towers in which communication will be available for each area.

It is proposed to solve the problem using a fuzzy graph.

We are building a fuzzy graph showing the proximity of the districts to each other. Proximity indicates the strength of the signal that will be in
the neighboring area if a tower is installed in the current one.

1. There is an existing set of towers. It is checked whether this set of towers is sufficient to cover the entire
area under consideration.

2. We are looking for a minimum subgraph of areas in which it is necessary to install towers to fully
cover the area.

3. We are looking for a minimal subgraph, only indicating the minimum signal strength that should be in each area.

"""

from fuzzyops.graphs.fuzzgraph.graph import FuzzyGraph
from fuzzyops.graphs.algorithms.dominating import dominating_set, fuzzy_dominating_set, is_dominating
from fuzzyops.graphs.algorithms.factoring import mle_clusterization_factors
from fuzzyops.graphs.algorithms.transport import shortest_path

######################################################
# 1. Checking the dominant subset
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

print(f"Is the subset {check_set} dominant: {'yes' if isDom else 'no'}")

######################################################
# 2. Defining the dominant subset
######################################################


ds = dominating_set(graph_zones)

print(f"The dominant subset: {ds}")

######################################################
# 3. Definition of a fuzzy dominant subset
######################################################


fd = fuzzy_dominating_set(graph_zones, [3, 1, 1])

print(f"Fuzzy Dominant Set: {fd}")

"""
Task:
segmentation of clients by groups

There is evidence between some customers about the similarity of their behavior, how close or far they are from
each other. To conduct A/B tests, it is necessary to divide users into two clusters based on their similarity. 

We build a fuzzy graph, and indicate the similarity between the clients. Graph nodes are clients, graph connections are their similarity. 
After that, we apply the mle_clusterization_factors method, specifying the required number of clusters.
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

print(f"Splitting into clusters: {clusters}")

"""
Task:
Solving a logistical problem

It is necessary to deliver the cargo from point A to point B in the shortest possible time, while taking into account the possibility of traffic jams or other
factors affecting the travel time. 

We are building a fuzzy graph that displays a road map, indicating on the links the fuzzy time to overcome certain
road sections. Fuzziness means likely traffic jams, traffic lights, repairs, or other factors that
may affect travel time.

After that, we apply the shortest_path function, specifying the start point and the end point. After applying the function, 
a list of nodes is displayed, which should be traversed to get from point A to point B in the shortest possible time.

"""

######################################################
# 5. Determining the optimal path
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

print(f"The shortest path: {path}")
