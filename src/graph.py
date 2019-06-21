import abc
import csv
import math


class _Node(abc.ABC):
    # TODO: re-think this as ADT
    """ADT to represent abstract concept of a node.

    Methods to be implemented by sublcasses:
    ---
        distance_to(self, other)
            Returns distance to other node.
    """
    # declare empty slots to allow subclasses to have their own slots
    __slots__ = []

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    # this is all the user needs to know about the node
    def distance_to(self, other):
        """Returns the distance from this node to other node of same type.

        If distance to other node is 0, returns 'inf' instead.

        Args
        ---
            other: an instance of the same class

        Returns
        ---
            distance: float
        """
        pass


class _Point2D(_Node):
    # TODO: re-think point as ADT
    """Lightweight ADT to help represent a 2D point node.

    Methods
    ---
        __init__(self, x, y)
            Initialize new _Poin2D node with provided coordinates.

        distance_to(self, other)
            See base class.
    """

    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = self._to_float(x)
        self.y = self._to_float(y)

    def _to_float(self, value):
        """Attempts to convert value to float, if possible."""

        try:
            x = float(value)
        except:
            raise TypeError(
                'Supplied value cannot be converted to float.')

        return x

    def distance_to(self, other):
        """See base class."""

        # preconditions
        if type(other) is not type(self):
            raise TypeError('other is not the same node type.')

        distance = self._distance_euclidean_to(other)
        if distance == 0:
            distance = float('inf')  # distance to itself is inf

        # postconditions
        assert type(distance) == float
        assert distance > 0

        return distance

    def _distance_euclidean_to(self, other):
        """Returns euclidean distance between this and other _Point2D."""

        # bad abstraction break by knowing that other point has x and y
        x_displacement_sq = (other.x - self.x)**2
        y_displacement_sq = (other.y - self.y)**2

        return math.sqrt(x_displacement_sq + y_displacement_sq)


class Graph:
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
        # dict should be non-empty
        if len(node_dict) == 0:
            raise ValueError('node_dict is empty.')

        # values must be instances of _Node
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
            raise KeyError(
                'Either source or destination nodes are not in the graph.')

        # this is potentially bad
        return source_node.distance_to(destination_node)


def read_csv(path):

    with open(path) as csvfile:
        reader = csv.reader(csvfile)

        # skip first 3 lines - name, comment, type
        for _ in range(3):
            next(reader)

        # DIMENSION: 4 tells how many nodes our graph will have
        row = next(reader)[0]
        key, value = row.split(':')
        assert key.strip() == 'DIMENSION'
        n_nodes = int(value.strip())

        # EDGE_WEIGHT_TYPE: EUC_2D tells us the type of an edge we operate on
        row = next(reader)[0]
        key, _ = row.split(':')
        assert key.strip() == 'EDGE_WEIGHT_TYPE'

        # NODE_COORD_SECTION row signifies the beginning of data body
        row = next(reader)[0]
        assert row.strip() == 'NODE_COORD_SECTION'

        # next n_nodes lines are nodes: first digit is id, next two are coords
        node_dict = {}

        for _ in range(n_nodes):
            row = next(reader)[0]
            node_id, *coordinates = row.split()
            node_dict[node_id] = _Point2D(*coordinates)

        # postcondition
        row = next(reader)[0]
        assert row.strip() == 'EOF'

    graph = Graph(node_dict)

    return graph
