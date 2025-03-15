# импорт класса нечеткого графа
from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from fuzzyops.sequencing_assignment import FuzzySASolver

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
            edge_number_eq_type='base',
        )

solver = FuzzySASolver()
solver.load_graph(graph)

# Число задач и число работников одинаково
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

# создаем двудольный граф и назначаем ребру стоимость как нечеткое значение
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
# результат - какому работнику какую задачу назначить и итоговая стоимость
print(result)