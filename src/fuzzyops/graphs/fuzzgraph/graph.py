


from .numbers import GraphTriangleFuzzyNumber

from .edges import GraphDirectedEdge, GraphUndirectedEdge

from .nodes import GraphSimpleNode

from typing import Union, List

node_types = Union['min', 'base', 'max']


class FuzzyGraph:

    def __init__(
            self,
            node_type: str = 'simple',
            node_number_type: str = 'triangle',
            edge_type: str = 'undirected',
            edge_number_type: str = 'triangle',
            node_number_math_type: node_types = None,
            node_number_eq_type: node_types = None,
            edge_number_math_type: node_types = None,
            edge_number_eq_type: node_types = None,

    ):
        """
        Класс для представления нечеткого графа.

        Тип операций (node_number_math_type, edge_number_math_type) для заданных нечетких чисел, если операции могут
        быть выполнены разными способами. Например, для треугольных чисел, основания складываются или вычитаются по
        правилам обычной алгебры, а границы могут вычисляться следующими способами: 'mean' (для верхней и нижней границ
        результата вычисляется среднее значение соответственно верхних и нижних границ двух изначальных чисел),
        'min' (верхней и нижней границами результата становятся минимальные значения соответственно верхних и нижних
        границ двух изначальных чисел), 'max' (верхней и нижней границами результата становятся максимальные значения
        соответственно верхних и нижних границ двух изначальных чисел) и 'sum' (для верхней и нижней границ результата
        вычисляется сумма соответственно верхних и нижних границ двух изначальных чисел);

        Тип для сравнения нечетких чисел (node_number_eq_type, edge_number_eq_type), если для данных чисел могут быть
        применены разные способы сравнения. Например, для треугольных чисел, возможны три способа – 'bas'e (треугольные
        числа сравниваются между собой по их основанию), 'min' (треугольные числа сравниваются между собой по их нижней
        границе) и 'max' (треугольные числа сравниваются между собой по их нижней границе);

        Attributes:
            _nodes (dict): Словарь узлов графа, где ключ - индекс узла, значение - объект узла.
            _edges (list): Список ребер графа.
            _node_number_class: Класс числа для описания значений узлов.
            _edge_number_class: Класс числа для описания значений ребер.
            _edge_class: Класс для представления ребер (направленных или ненаправленных).
            _node_type (str): Тип узлов графа (например, простой).
            _node_params (dict): Параметры для создания чисел узлов.
            _edge_params (dict): Параметры для создания чисел ребер.

        Args:
            node_type (str): Тип узлов ('simple' по умолчанию).
            node_number_type (str): Тип чисел для узлов ('triangle' по умолчанию).
            edge_type (str): Тип ребер ('undirected' по умолчанию).
            edge_number_type (str): Тип чисел для ребер ('triangle' по умолчанию).
            node_number_math_type (node_types): Тип операций для нечетких чисел узлов.
            node_number_eq_type (node_types): Тип для сравнения нечетких чисел узлов.
            edge_number_math_type (node_types): Тип операций для нечетких числах ребер.
            edge_number_eq_type (node_types): Тип для сравнения нечетких чисел ребер.

        Methods:
            get_nodes_amount() -> int:
                Возвращает количество узлов в графе.

            get_edges_amount() -> int:
                Возвращает количество ребер в графе.

            add_node(value=None) -> None:
                Добавляет узел в граф.

            add_edge(from_ind: int, to_ind: int, value: List[int]) -> None:
                Добавляет ребро между двумя узлами.

            check_node(index: int) -> bool:
                Проверяет, существует ли узел с заданным индексом.

            get_directly_connected(index: int) -> List[int]:
                Получает список узлов, которые напрямую связаны с заданным узлом.

            get_stronger_directly_connected(index: int, value: List[int]) -> List[int]:
                Получает список узлов, которые напрямую связаны с заданным узлом и имеют более сильные ребра.

            check_directed_edge(from_ind: int, to_ind: int) -> bool:
                Проверяет, существует ли направленное ребро между двумя узлами.

            get_edge_len(from_ind: int, to_ind: int) -> int:
                Получает длину ребра между двумя узлами.

            get_adjacency_matrix() -> List[List[int]]:
                Возвращает матрицу смежности графа.

            check_nodes_full(nodes: List[int]) -> bool:
                Проверяет, охватывают ли указанные узлы все узлы в графе.

            get_nodes_list() -> List[int]:
                Получает список индексов всех узлов в графе.
        """

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
    ) -> int:
        """
       Возвращает количество узлов в графе.

       Returns:
           int: Количество узлов.
        """

        return len(self._nodes)

    def get_edges_amount(
            self
    ) -> int:
        """
        Возвращает количество ребер в графе.

        Returns:
            int: Количество ребер.
        """

        return len(self._edges)

    def add_node(
            self,
            value=None,
    ) -> None:
        """
        Добавляет узел в граф.

        Args:
            value: Значение для узла (по умолчанию None). Если указано, создается нечеткое число для узла.
        """

        ind = len(self._nodes)
        if not (value is None):
            value = self._node_number_class(value, **self._node_params)
        node = GraphSimpleNode(ind, value)
        self._nodes[ind] = node

    def add_edge(
            self,
            from_ind: int,
            to_ind: int,
            value: List[int],
    ):
        """
        Добавляет ребро между двумя узлами.

        Args:
            from_ind (int): Индекс исходного узла.
            to_ind (int): Индекс целевого узла.
            value (List[int]): Значение для ребра нечеткое число - массив из трех чисел для треугольного числа.

        Raises:
            Exception: Исключение возникает, если граф не допускает петель или если указаны несуществующие узлы.
        """

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
            index: int
    ) -> bool:
        """
        Проверяет, существует ли узел с заданным индексом.

        Args:
            index (int): Индекс узла.

        Returns:
            bool: True, если узел существует, иначе False.
        """

        return index in self._nodes.keys()

    def get_directly_connected(
            self,
            index: int
    ) -> List[int]:
        """
        Получает список узлов, которые напрямую связаны с заданным узлом.

        Args:
            index (int): Индекс узла.

        Returns:
            List[int]: Список индексов напрямую связанных узлов.

        Raises:
            Exception: Исключение возникает, если узел не существует.
        """

        if index not in self._nodes.keys():
            raise Exception('no such node')
        return self._nodes[index].get_outcome_edges()

    def get_stronger_directly_connected(
            self,
            index: int,
            value: List[int]
    ) -> List[int]:
        """
        Получает список узлов, которые напрямую связаны с заданным узлом и имеют более сильные ребра.

        Args:
            index (int): Индекс узла.
            value (List[int]): Список значений, по которым осуществляется фильтрация.

        Returns:
            List[int]: Список индексов более сильно связанных узлов.

        Raises:
            Exception: Исключение возникает, если узел не существует.
        """

        if index not in self._nodes.keys():
            raise Exception('no such node')
        return self._nodes[index].get_outcome_stronger_edges(value)

    def check_directed_edge(
            self,
            from_ind: int,
            to_ind: int,
    ) -> bool:
        """
        Проверяет, существует ли направленное ребро между двумя узлами.

        Args:
            from_ind (int): Индекс исходного узла.
            to_ind (int): Индекс целевого узла.

        Returns:
            bool: True, если направленное ребро существует, иначе False.

        Raises:
            Exception: Исключение возникает, если узлы не существуют.
        """

        if (self._node_type != 'looped') and (from_ind == to_ind):
            return False
        if from_ind not in self._nodes.keys():
            raise Exception('no such node')
        if to_ind not in self._nodes.keys():
            raise Exception('no such node')
        return self._nodes[from_ind].check_is_directly_connected(to_ind)

    def get_edge_len(
            self,
            from_ind: int,
            to_ind: int,
    ) -> int:
        """
        Получает длину ребра между двумя узлами.

        Args:
            from_ind (int): Индекс исходного узла.
            to_ind (int): Индекс целевого узла.

        Returns:
            int: Длина ребра между узлами.

        Raises:
            Exception: Исключение возникает, если граф не допускает петель или если указаны несуществующие узлы.
        """

        if (self._node_type != 'looped') and (from_ind == to_ind):
            raise Exception('graph is not looped')
        if from_ind not in self._nodes.keys():
            raise Exception('no such node')
        if to_ind not in self._nodes.keys():
            raise Exception('no such node')

        return self._nodes[from_ind].get_len_to(to_ind)

    def get_adjacency_matrix(
            self
    ) -> List[List[int]]:
        """
        Возвращает матрицу смежности графа.

        Матрица смежности - это квадратная матрица, где элемент (i, j) представляет длину ребра
        между узлом i и узлом j. Если ребра нет, элемент будет равен None.

        Returns:
            List[List[int]]: Матрица смежности графа.

        Note:
            Если граф направленный, матрица будет отображать направление ребер.
            Если узлы не связаны, соответствующие элементы будут равны None.
        """

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
            nodes: List[int]
    ) -> bool:
        """
        Проверяет, охватывают ли указанные узлы все узлы в графе.

        Args:
            nodes (List[int]): Список индексов узлов для проверки.

        Returns:
            bool: True, если указанные узлы охватывают все узлы в графе, иначе False.
        """

        for n in nodes:
            if not (n in self._nodes):
                return False

        for n in self._nodes:
            if not (n in nodes):
                return False

        return True

    def get_nodes_list(
            self
    ) -> List[int]:
        """
        Получает список индексов всех узлов в графе.

        Returns:
            List[int]: Список индексов узлов.
        """

        return [i for i in self._nodes]
