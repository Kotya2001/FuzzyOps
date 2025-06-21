from typing import List, Union
from fuzzyops.fuzzy_numbers import FuzzyNumber


class Node:
    """
    Represents a node in a fuzzy analytical network

    Attributes:
        name (str): Node name
        in_edges (List[Edge]): The list of incoming edges for this node
        out_edges (List[Edge]): The list of outgoing edges from this node

    Args:
        name (str): Node name
    """
    def __init__(self, name: str):
        self.name = name
        self.in_edges = []
        self.out_edges = []

    def add_in_edge(self, edge: ['Edge']) -> None:
        """
        Adds an incoming edge to a node

        Args:
            edge (Edge): Edge
        """
        self.in_edges.append(edge)

    def add_out_edge(self, edge: ['Edge']) -> None:
        """
        Adds an outgoing edge from the node

        Args:
            edge (Edge): Edge
        """
        self.out_edges.append(edge)


class Edge:
    """
    Represents an edge in a fuzzy analytical network

    Attributes:
        start_node (Node): The initial node of the edges
        end_node (Node): The end node of the edge
        weight (float): The weight of the edge, representing its degree of feasibility

    Args:
        start_node (Node): The initial node of the edge
        end_node (Node): The end node of the edge.
        weight (float): The weight of the edge, representing its degree of feasibility
    """
    def __init__(self, start_node: Node, end_node: Node, weight: float):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight
        self.start_node.add_out_edge(self)
        self.end_node.add_in_edge(self)


class Graph:
    """
    It represents a directed graph, a fuzzy analytical network
    The algorithm is implemented according to the article
    https://cyberleninka.ru/article/n/nechetkaya-alternativnaya-setevaya-model-analiza-i-planirovaniya-proekta-v-usloviyah-neopredelennosti

    Attributes:
        nodes (dict): Dictionary of nodes in a graph
        edges (List[Edge]): A list of edges in a graph
    """

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_name: str) -> Node:
        """
        Adds a node to the graph and returns it

        Args:
            node_name (str): The name of the initial node
        """
        if node_name not in self.nodes:
            new_node = Node(node_name)
            self.nodes[node_name] = new_node
        return self.nodes[node_name]

    def add_edge(self, start_node_name: Node, end_node_name: Node, weight: float) -> None:
        """
        Adds an edge to the graph

        Args:
            start_node_name (Node): The name of the initial node
            end_node_name (Node): Destination Node Name
            weight (float): Weight
        """
        start_node = self.add_node(start_node_name)
        end_node = self.add_node(end_node_name)
        new_edge = Edge(start_node, end_node, weight)
        self.edges.append(new_edge)

    def get_paths_from_to(self, start_node_name: Node, end_node_name: Node) -> List[Node]:
        """
        Returns a list of all possible paths from the start node to the end node

        Args:
            start_node_name (Node): The name of the initial node
            end_node_name (Node): Destination Node Name

        Returns:
            List[Node]: A list of paths representing lists of node names
        """
        paths = []
        stack = [(start_node_name, [])]
        while stack:
            current_node, path = stack.pop()
            if current_node == end_node_name:
                paths.append(path + [current_node])
            else:
                for edge in self.nodes[current_node].out_edges:
                    if edge.end_node.name not in path:
                        stack.append((edge.end_node.name, path + [current_node]))
        return paths

    def calculate_path_fuzziness(self, path: List[Node]) -> float:
        """
        Calculates the fuzziness of a given path

        Args:
            path (List[Node]): A path represented as a list of node names

        Returns:
            float: Estimation of path fuzziness

        Raises:
            ValueError: If the path is invalid (i.e. there are no edges between nodes)
        """

        fuzziness = 1.0
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            edges = [edge for edge in self.edges if
                     edge.start_node.name == start_node and edge.end_node.name == end_node]
            if len(edges) == 0:
                raise ValueError("Path is invalid")
            min_weight = min([edge.weight for edge in edges])
            fuzziness *= min_weight
        return fuzziness

    def find_most_feasible_path(self, start_node_name: Node, end_node_name: Node) -> List[str]:
        """
        Finds the most feasible path between two nodes based on fuzziness

        Args:
            start_node_name (Node): The name of the initial node
            end_node_name (Node): The name of the destination node

        Returns:
            List[str]: The most feasible path is represented as a list of node names
        """

        paths = self.get_paths_from_to(start_node_name, end_node_name)
        feasible_paths = [(path, self.calculate_path_fuzziness(path)) for path in paths]
        sorted_paths = sorted(feasible_paths, key=lambda x: x[1], reverse=True)
        return sorted_paths[0][0] if sorted_paths else None

    def macro_algorithm_for_best_alternative(self) -> Union[List[str], float]:
        """
        Performs a macro algorithm to determine the best alternative in the network model

        Returns:
            Union[List[str], float]: The best alternative path and its feasibility assessment
        """

        best_alternative = None
        max_feasibility = 0.0
        for start_node in self.nodes.values():
            for end_node in self.nodes.values():
                if start_node != end_node:
                    most_feasible_path = self.find_most_feasible_path(start_node.name, end_node.name)
                    if most_feasible_path:
                        path_feasibility = self.calculate_path_fuzziness(most_feasible_path)
                        if path_feasibility > max_feasibility:
                            max_feasibility = path_feasibility
                            best_alternative = most_feasible_path
        return best_alternative, max_feasibility


def calc_final_scores(f_nums: List[FuzzyNumber]) -> float:
    """
    Calculates the final score from a list of fuzzy numbers

    Args:
        f_nums (List[FuzzyNumber]): List of fuzzy numbers

    Returns:
        float: The result of defuzzification of fuzzy numbers
    """

    res = 1
    for f_num in f_nums:
        res *= f_num
    return res.defuzz()
