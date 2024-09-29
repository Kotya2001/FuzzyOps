"""
class of directed edge.

undirected edge goes from one node to other
"""

from .base_edge import GraphBaseEdge

class GraphDirectedEdge(GraphBaseEdge):

    def is_going_from_node(self, index) -> bool:
        if index == self._from:
            return True
        return False

    def is_going_to_node(self, index) -> bool:
        if index == self._to:
            return True
        return False

    def get_connected_nodes(self) -> list:
        return [self._from, self._to]

    def get_from_nodes(self) -> list:
        return [self._from]

    def get_to_nodes(self) -> list:
        return [self._to]