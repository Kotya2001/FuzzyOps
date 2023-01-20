"""
class of undirected edge.

undirected edge goes from both connected nodes and goes to both
"""

from .base_edge import GraphBaseEdge

class GraphUndirectedEdge(GraphBaseEdge):

    def is_going_from_node(self, index) -> bool:
        if index == self._from or index == self._to:
            return True
        return False

    def is_going_to_node(self, index) -> bool:
        if index == self._from or index == self._to:
            return True
        return False

    def get_connected_nodes(self) -> list:
        return [self._from, self._to]

    def get_from_nodes(self) -> list:
        return [self._from, self._to]

    def get_to_nodes(self) -> list:
        return [self._from, self._to]

