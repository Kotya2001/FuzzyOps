from fuzzyops.graphs.fuzzgraph import FuzzyGraph
from typing import List, Dict


class FuzzySASolver:
    """
    Represents the Assignment Problem solver (SAS) using a fuzzy graph

    Attributes:
        _graph (FuzzyGraph): A fuzzy graph containing information about employees and tasks
        _workers (List[str]): List of employees
        _tasks (List[str]): Task list
    """

    def __init__(self):
        self._graph = None
        self._workers = None
        self._tasks = None

    def load_graph(self, graph: FuzzyGraph) -> None:
        """
        Loads an empty graph with a certain fuzzy mathematical type

        Args:
            graph (FuzzyGraph): Fuzzy graph for loading

        Raises:
            Exception: If the graph is no longer empty or is not an instance of FuzzyGraph
        """

        if not (type(graph) is FuzzyGraph):
            raise Exception('Can use only FuzzGraph')

        if (graph.get_nodes_amount() != 0) or (graph.get_edges_amount() != 0):
            raise Exception('Can load only empty graph')

        self._graph = graph

    def load_tasks_workers(self, tasks: List[str], workers: List[str]) -> None:
        """
        Loads lists of tasks and employees

        Args:
            tasks (List[str]): The list of tasks to download
            workers (List[str]): List of workers to upload

        Raises:
            Exception: If the graph is not loaded, or if tasks and workers are not lists
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

    def load_task_worker_pair_value(self, task: str, worker: str, value: List[int]) -> None:
        """
        Loads the cost of an assignment for a pair of employee and task

        Args:
            task (str): Assignment task
            worker (str): The employee to whom the task is assigned
            value (List[int]): The cost of the appointment

        Raises:
            Exception: If the list of tasks or workers is not loaded
        """
        if (self._workers is None) or (self._tasks is None):
            raise Exception('Upload tasks and workers lists with `load_tasks_workers` function')
        if not (task in self._tasks):
            raise Exception('No such task')
        if not (worker in self._workers):
            raise Exception('No such worker')

        w = self._workers.index(worker)
        t = self._tasks.index(task) + len(self._workers)

        self._graph.add_edge(w, t, value)

    def solve(self) -> Dict:
        """
        The main function of the solution, which implements the Hungarian algorithm

        Returns:
            Dict: A dictionary with employee assignments to tasks and the total cost

        Raises:
            Exception: If the graph or the lists of workers/tasks are not loaded
        """
        if self._graph is None:
            raise Exception('There is no graph loaded to solver, use `load_graph` function')
        if (self._workers is None) or (self._tasks is None):
            raise Exception('Upload tasks and workers lists with `load_tasks_workers` function')

        # Creating an array for storing assignments (employee indexes)
        assignment = [-1] * len(self._workers)

        # Creating an array to track busy tasks
        occupied_tasks = [-1] * len(self._tasks)

        # Iteratively improving assignments
        for worker in range(len(self._workers)):
            # List of available tasks to assign
            available_tasks = [task + len(self._workers) for task in range(len(self._tasks)) if
                               occupied_tasks[task] == -1]

            # Trying to find an improvement for the current employee
            for task in available_tasks:
                try:
                    cost = self._graph.get_edge_len(worker, task)
                except:
                    continue
                if assignment[worker] == -1 or cost < self._graph.get_edge_len(worker, assignment[worker]):
                    # We release the previous task, if it was
                    if assignment[worker] != -1:
                        occupied_tasks[assignment[worker] - len(self._workers)] = -1
                    # Assigning a new task
                    assignment[worker] = task
                    occupied_tasks[task - len(self._workers)] = worker

        # Calculating the total cost
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
