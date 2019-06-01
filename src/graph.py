import abc
import csv
import math


class _Node(abc.ABC):
    # TODO: re-think this as ADT
    """ADT to represent abstract concept of a node.

    Methods to be implemented by sublcasses
    ---
        distance_to(self, other)
            Returns distance to other node
    """
    # declare empty slots to allow subclasses to have their own slots
    __slots__ = []

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    # this is all the user needs to know about the node
    def distance_to(self, other):
        pass


class _Point2D(_Node):
    # TODO: re-think point as ADT
    """Lightweight ADT to help represent a 2D point node.

    Methods
    ---
        __init__(self, x, y)
            Initialize new _Poin2D node with provided coordinates.

        distance(self, other)
            Returns distance between this and other node of the same type"""

    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = self._to_float(x)
        self.y = self._to_float(y)

    def _to_float(coord):
        # attempt to convert supplied coordinate to float
        try:
            x = float(coord)
        except:
            print('Supplied coordinates cannot be converted to float.')
            raise TypeError

        return x

    def distance_to(self, other):
        """Returns the distance from this node to other node of same type.

        If distance to other node is 0, returns 'inf' instead.
        """

        if type(other) is not type(self):
            raise TypeError('other is not the same node type')

        distance = self._distance_euclidean_to(other)
        # convention for graphs -- distance to itself is inf
        if distance == 0:
            distance = float('inf')

        return distance

    def _distance_euclidean_to(self, other):
        # return square of euclidean dist between this and other Point2D

        # bad abstraction break by knowing that other point has x and y
        x_displacement_sq = (other.x - self.x)**2
        y_displacement_sq = (other.y - self.y)**2

        return math.sqrt(x_displacement_sq + y_displacement_sq)


class Graph:
    # TODO: unit tests, re-think the graph as ADT
    """Graph Abstract Data Type.

    Methods
    ---
        nodes(self)
            Returns an iterator over the nodes of this graph.

        distance(self, source, destination)
            Returns distance between source and destination nodes.
    """

    def __init__(self, node_dict):
        # low-level representation of graph
        # node is represented by key-val pair {id: _Node}
        self._validate(node_dict)

        self._node_dict = node_dict

    def _validate(self, node_dict):
        if len(node_dict) == 0:
            raise ValueError('node_dict is empty.')

        for val in node_dict.values():
            if not isinstance(val, _Node):
                raise TypeError('Provided values are not nodes.')

    def nodes(self):
        """Returns an iterator over the nodes of this graph."""

        for node in self._node_dict.keys():
            yield node

    def distance(self, source, destination):
        """Returns distance between source and destination nodes."""

        try:
            source_node = self._node_dict[source]
            destination_node = self._node_dict[destination]
        except:
            print('Either source or destination nodes are not in the graph.')
            raise KeyError

        return source_node.distance_to(destination_node)


def read_csv(path):

    with open(path) as csvfile:
        reader = csv.reader(csvfile)

        # skip first 3 lines - name, comment, type
        for _ in range(3):
            next(reader)

        # DIMENSION: 4 tells how many nodes our graph will have
        row = next(reader)[0]
        _, value = row.split(':')
        n_nodes = int(value.strip())

        # EDGE_WEIGHT_TYPE: EUC_2D tells us the type of an edge we operate on
        row = next(reader)[0]
        _, value = row.split(':')
        edge_weight_type = value.strip()

        # NODE_COORD_SECTION row signifies the beginning of data body
        row = next(reader)[0]
        assert(row == 'NODE_COORD_SECTION')

        # next n_nodes lines are nodes: first digit is id, next two are coords
        node_dict = {}
        for _ in range(n_nodes):
            row = next(reader)[0]
            node_id, *coordinates = row.split()
            node_dict[node_id] = _Point2D(*coordinates)

        assert(next(reader)[0] == 'EOF')

    graph = Graph(node_dict)

    return graph
