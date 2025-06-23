"""
Task:

A parts manufacturing company receives orders from various distributors and must distribute these orders between
production lines. However, the time or cost of production may fluctuate due to various factors such
as the availability of resources, the workload of the equipment and the skill level of the workers.

We have three distributors (distributor_1, distributor_2, distributor_3) who receive orders and need
to distribute them between production lines (production_1, production_2, production_3).
However, each distributor-production line pair has different order fulfillment characteristics
represented by fuzzy values. These values indicate the cost of completing a specific
distributor's order on a specific production line.

The goal is for the company to distribute orders in such a way as to minimize the total time/cost of completion, given
the uncertainty in estimates.

To do this, a bipartite fuzzy graph is created in which the cost for each distributor-production line pair is indicated

"""

from fuzzyops.graphs.fuzzgraph.graph import FuzzyGraph
from fuzzyops.sequencing_assignment.solver import FuzzySASolver


graph = FuzzyGraph(
    node_number_math_type='min',
    node_number_eq_type='max',
    edge_number_math_type='mean',
    edge_number_eq_type='base',
)

solver = FuzzySASolver()
solver.load_graph(graph)

solver.load_tasks_workers(
    [
        'distributor_1',
        'distributor_2',
        'distributor_3'
    ],
    [
        'production_1',
        'production_2',
        'production_3'
    ]
)

solver.load_task_worker_pair_value('distributor_1', 'production_1', [6, 1, 2])
solver.load_task_worker_pair_value('distributor_1', 'production_2', [3, 1, 1])
solver.load_task_worker_pair_value('distributor_1', 'production_3', [2, 1, 2])
solver.load_task_worker_pair_value('distributor_2', 'production_1', [4, 2, 1])
solver.load_task_worker_pair_value('distributor_2', 'production_2', [5, 1, 1])
solver.load_task_worker_pair_value('distributor_2', 'production_3', [1, 1, 1])
solver.load_task_worker_pair_value('distributor_3', 'production_1', [2, 1, 1])
solver.load_task_worker_pair_value('distributor_3', 'production_2', [3, 2, 1])
solver.load_task_worker_pair_value('distributor_3', 'production_3', [2, 1, 2])

result = solver.solve()

print(result)