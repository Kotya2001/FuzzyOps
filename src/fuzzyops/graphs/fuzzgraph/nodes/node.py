"""
graph node class

every node have inner value and list of all connected edges
"""

class GraphSimpleNode:
    def __init__(
            self,
            index,
            value=None,
    ):
        """

        :param index: self index into graph
        :param value: inner value - fuzzy number
        """
        self._index = index
        self._edges = []

    def add_edge(self, edge):
        if edge.is_connected_to_node(self._index):
            self._edges.append(edge)
        else:
            raise Exception('new edge is not connected to this node')


    def check_is_directly_connected(self, to_index):
        for e in self._edges:
            if e.is_going_to_node(to_index):
                return True
        return False


    def get_outcome_edges(self):
        nodes = set()
        for edge in self._edges:
            nd = edge.get_to_nodes()
            for n in nd:
                nodes.add(n)
        if self._index in nodes:
            nodes.remove(self._index)
        return list(nodes)


    def get_outcome_stronger_edges(self, value):
        nodes = set()
        for edge in self._edges:
            if edge.is_stronger(value):
                nd = edge.get_to_nodes()
                for n in nd:
                    nodes.add(n)
        if self._index in nodes:
            nodes.remove(self._index)
        return list(nodes)


    def get_len_to(self, to_index):
        for e in self._edges:
            if e.is_going_to_node(to_index):
                return e.get_value()
        raise Exception("node dont connected to given node")


