# importing a fuzzy graph class
from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.sequencing_assignment import FuzzySASolver

# creating a fuzzy graph of an undirected graph (edge_type parameter = 'undirected')
# with node_number_math_type (type of calculation operations for vertices - 'min'),
# with node_number_eq_type (type of comparison operations for vertices - 'max')
# with edge_number_math_type (type of calculation operations for edges - 'mean'),
# with edge_number_eq_type (type of comparison operations for edges - 'base')
# the graph is built on triangular numbers
graph = FuzzyGraph(
            node_number_math_type='min',
            node_number_eq_type='max',
            edge_number_math_type='mean',
            edge_number_eq_type='base',
        )

solver = FuzzySASolver()
solver.load_graph(graph)

# The number of tasks and the number of employees are the same
solver.load_tasks_workers(
    [
        'task_A',
        'task_B',
        'task_C'
    ],
    [
        'worker_X',
        'worker_Y',
        'worker_Z'
    ]
)

# we create a bipartite graph and assign a cost to an edge as a fuzzy value
solver.load_task_worker_pair_value('task_A', 'worker_X', [6, 1, 2])
solver.load_task_worker_pair_value('task_A', 'worker_Y', [3, 1, 1])
solver.load_task_worker_pair_value('task_A', 'worker_Z', [2, 1, 2])
solver.load_task_worker_pair_value('task_B', 'worker_X', [4, 2, 1])
solver.load_task_worker_pair_value('task_B', 'worker_Y', [5, 1, 1])
solver.load_task_worker_pair_value('task_B', 'worker_Z', [1, 1, 1])
solver.load_task_worker_pair_value('task_C', 'worker_X', [2, 1, 1])
solver.load_task_worker_pair_value('task_C', 'worker_Y', [3, 2, 1])
solver.load_task_worker_pair_value('task_C', 'worker_Z', [2, 1, 2])

result = solver.solve()
# the result is which employee gets which task and the total cost
print(result)