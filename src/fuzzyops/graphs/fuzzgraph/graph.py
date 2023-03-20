from .numbers import *

from .edges import *

from .nodes import GraphSimpleNode



class FuzzyGraph:
    """
    todo(comments)
    """
    def __init__(
        self,
        node_type='simple',
        node_number_type='triangle',
        edge_type='undirected',
        edge_number_type='triangle',
        node_number_math_type=None,
        node_number_eq_type=None,
        edge_number_math_type=None,
        edge_number_eq_type=None,

    ):
        self._nodes = {}
        self._edges = []

        if node_number_type == 'triangle':
            self._node_number_class = GraphTriangleFuzzyNumber
        else:
            raise Exception('wrong number type')

        if edge_number_type == 'triangle':
            self._edge_number_class = GraphTriangleFuzzyNumber
        else:
            raise Exception('wrong number type')

        if edge_type == 'undirected':
            self._edge_class = GraphUndirectedEdge
        elif edge_type == 'directed':
            self._edge_class = GraphDirectedEdge

        self._node_type = 'simple'

        self._node_params = {'eq_type': node_number_eq_type, 'math_type': node_number_math_type}
        self._edge_params = {'eq_type': edge_number_eq_type, 'math_type': edge_number_math_type}


    def get_nodes_amount(
        self
    ):
        return len(self._nodes)


    def add_node(
        self,
        value=None,
    ):
        ind = len(self._nodes)
        if not(value is None):
            value = self._node_number_class(value, **self._node_params)
        node = GraphSimpleNode(ind, value)
        self._nodes[ind] = node


    def add_edge(
        self,
        from_ind,
        to_ind,
        value,
    ):
        if (self._node_type != 'looped') and (from_ind == to_ind):
            raise Exception('graph is not looped')
        if from_ind not in self._nodes.keys():
            raise Exception('no such node')
        if to_ind not in self._nodes.keys():
            raise Exception('no such node')

        value = self._edge_number_class(value, **self._edge_params)

        edge = self._edge_class(weight=value, from_node=from_ind, to_node=to_ind)

        self._nodes[from_ind].add_edge(edge)
        self._nodes[to_ind].add_edge(edge)
        self._edges.append(edge)


    def check_node(
        self,
        index
    ):
        return index in self._nodes.keys()


    def get_directly_connected(
        self,
        index
    ):
        if index not in self._nodes.keys():
            raise Exception('no such node')

        return self._nodes[index].get_outcome_edges()

    def get_stronger_directly_connected(
        self,
        index,
        value
    ):
        if index not in self._nodes.keys():
            raise Exception('no such node')

        return self._nodes[index].get_outcome_stronger_edges(value)


    def check_directed_edge(
        self,
        from_ind,
        to_ind,
    ):
        if (self._node_type != 'looped') and (from_ind == to_ind):
            return False
        if from_ind not in self._nodes.keys():
            raise Exception('no such node')
        if to_ind not in self._nodes.keys():
            raise Exception('no such node')
        return self._nodes[from_ind].check_is_directly_connected(to_ind)


    def get_edge_len(
        self,
        from_ind,
        to_ind,
    ):
        if (self._node_type != 'looped') and (from_ind == to_ind):
            raise Exception('graph is not looped')
        if from_ind not in self._nodes.keys():
            raise Exception('no such node')
        if to_ind not in self._nodes.keys():
            raise Exception('no such node')

        return self._nodes[from_ind].get_len_to(to_ind)


    def get_adjacency_matrix(
        self
    ):
        matrix = []
        for from_ind in range(len(self._nodes)):
            row = []
            for to_ind in range(len(self._nodes)):
                num = None
                if ((self._node_type != 'looped') and (from_ind == to_ind)) or (from_ind != to_ind):
                    if self.check_directed_edge(from_ind, to_ind):
                        num = self.get_edge_len(from_ind, to_ind)
                row.append(num)
            matrix.append(row)
        return matrix


    def check_nodes_full(
        self,
        nodes
    ):
        for n in nodes:
            if not(n in self._nodes):
                return False

        for n in self._nodes:
            if not(n in nodes):
                return False

        return True


    def get_nodes_list(
        self
    ):
        return [i for i in self._nodes]