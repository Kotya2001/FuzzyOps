from src.fuzzyops.graphs.fuzzgraph.graph import FuzzyGraph
from src.fuzzyops.sequencing_assignment.solver import FuzzySASolver

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
