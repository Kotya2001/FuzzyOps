"""
base class of all edges.
"""

class GraphBaseEdge:
    def __init__(
            self,
            weight,
            from_node,
            to_node,
    ):
        self._weight = weight
        self._from = from_node
        self._to = to_node


    def get_value(self):
        return self._weight


    def is_stronger(self, value):
        return self._weight > value


    def is_connected_to_node(self, index) -> bool:
        """
        check if connected to node by any side of edge
        """
        if index == self._from or index == self._to:
            return True
        return False


    def is_going_from_node(self, index) -> bool:
        """
        check if edge goes from node
        """
        raise Exception('is not overwritten')


    def is_going_to_node(self, index) -> bool:
        """
        check if edge goes to node
        """
        raise Exception('is not overwritten')


    def get_connected_nodes(self) -> list:
        """
        get all attaches nodes
        """
        raise Exception('is not overwritten')


    def get_from_nodes(self) -> list:
        """
        get node which edge goes from
        """
        raise Exception('is not overwritten')


    def get_to_nodes(self) -> list:
        """
        get node which edge goes to
        """
        raise Exception('is not overwritten')
