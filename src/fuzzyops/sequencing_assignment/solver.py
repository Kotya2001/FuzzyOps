
from ..graphs.fuzzgraph import FuzzyGraph

class FuzzySASolver:
    def __init__(self):
        self._graph = None
        self._workers = None
        self._tasks = None


    def load_graph(self, graph):
        """
        loading empty graph with defined fuzzy math type
        """
        if not (type(graph) is FuzzyGraph):
            raise Exception('Can use only FuzzGraph')

        if (graph.get_nodes_amount() != 0) or (graph.get_edges_amount() != 0):
            raise Exception('Can load only empty graph')

        self._graph = graph



    def load_tasks_workers(self, tasks, workers):
        """
        load lists of workers and tasks
        """
        if not self._graph:
            raise Exception('Upload empty FuzzGraph with `load_graph` function')
        if not (type(tasks) is list):
            raise Exception('`tasks` can be only a list')
        if not (type(workers) is list):
            raise Exception('`workers` can be only a list')

        self._workers = workers
        self._tasks = tasks

        # workers
        for i in range(len(self._workers)):
            self._graph.add_node()

        # tasks
        for i in range(len(self._tasks)):
            self._graph.add_node()


    def load_task_worker_pair_value(self, task, worker, value):
        """
        load assignment cost to pair worker->task
        """
        if (self._workers is None) or (self._tasks is None):
            raise Exception('Upload tasks and workers lists with `load_tasks_workers` function')
        if not(task in self._tasks):
            raise Exception('No such task')
        if not(worker in self._workers):
            raise Exception('No such worker')

        w = self._workers.index(worker)
        t = self._tasks.index(task) + len(self._workers)

        self._graph.add_edge(w, t, value)


    def solve(self):
        """
        main solve function
        implementing Hungarian algorithm
        """
        if self._graph is None:
            raise Exception('There is no graph loaded to solver, use `load_graph` function')
        if (self._workers is None) or (self._tasks is None):
            raise Exception('Upload tasks and workers lists with `load_tasks_workers` function')


        # Создаем массив для хранения назначений (индексы работников)
        assignment = [-1] * len(self._workers)

        # Создаем массив для отслеживания занятых задач
        occupied_tasks = [-1] * len(self._tasks)

        # Итеративно улучшаем назначения
        for worker in range(len(self._workers)):
            # Список доступных задач для назначения
            available_tasks = [task + len(self._workers) for task in range(len(self._tasks)) if occupied_tasks[task] == -1]

            # Пытаемся найти улучшение для текущего работника
            for task in available_tasks:
                try:
                    cost = self._graph.get_edge_len(worker, task)
                except:
                    continue
                if assignment[worker] == -1 or cost < self._graph.get_edge_len(worker, assignment[worker]):
                    # Освобождаем предыдущую задачу, если она была
                    if assignment[worker] != -1:
                        occupied_tasks[assignment[worker]  - len(self._workers)] = -1
                    # Назначаем новую задачу
                    assignment[worker] = task
                    occupied_tasks[task - len(self._workers)] = worker


        # Вычисляем общую стоимость
        total_cost = None
        for worker in range(len(self._workers)):
            if assignment[worker] != -1:
                if total_cost is None:
                    total_cost = self._graph.get_edge_len(worker, assignment[worker])
                else:
                    total_cost += self._graph.get_edge_len(worker, assignment[worker])

        toRet = []
        for i in range(len(self._workers)):
            if assignment[i] != -1:
                toRet.append([self._workers[i], self._tasks[assignment[i] - len(self._workers)]])
            else:
                toRet.append([self._workers[i], 'no assignment'])

        return {
            'assignments': toRet,
            'cost': total_cost
        }
