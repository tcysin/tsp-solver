import csv
import math
import unittest

class _Node:
    # TODO: re-think this as ADT
    """ADT to represent abstract concept of a node.
    
    Methods to be implemented by sublcasses
    ---
        distance_to(self, other)
            Returns distance to other node
    """
    # declare empty slots to allow subclasses to have their own slots
    __slots__ = []

    # this is all a user needs to know about the node
    def distance_to(self, other):
        raise NotImplementedError('distance_to method should be implemented.')

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
        # TODO: think whether error handling is needed here
        self.x = x
        self.y = y

    def distance_to(self, other):
        """Returns the distance from this node to other node of same type.
        
        If distance to other node is 0, returns 'inf' instead.
        """

        distance = self._distance_euclidean_to(other)

        # distance to itself is 0 - we make it inf by convention
        if distance == 0:
            distance = float('inf')

        return distance

    # private methods
    def _distance_euclidean_to(self, other):
        # return square of euclidean dist between this and other Point2D 
        if type(other) is not type(self):
            raise TypeError('other node is not the same type')

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
        for val in node_dict.values():

            if not isinstance(val, _Node):
                raise TypeError('Provided values are not nodes.')
        
        self._node_dict = node_dict

    def nodes(self):
        """Returns an iterator over the nodes of this graph."""
        
        for node in self._node_dict.keys():
            yield node

    def distance(self, source, destination):
        """Returns distance between source and destination nodes."""

        source_node = self._node_dict[source]
        destination_node = self._node_dict[destination]

        return source_node.distance_to(destination_node)

def read_csv(path):

    with open(path) as csvfile:
        reader = csv.reader(csvfile)

        # skip first 3 lines - name, comment, type
        for _ in range(3):
            next(reader)

        # DIMENSION: 4 tells how many nodes our graph will have
        n_nodes = int(next(reader)[0].split(':')[1].strip())

        # EDGE_WEIGHT_TYPE: EUC_2D tells us the type of an edge we operate on
        node_type = next(reader)[0].split(':')[1].strip()

        # NODE_COORD_SECTION row signifies the beginning of data body
        assert(next(reader)[0] == 'NODE_COORD_SECTION')

        # next n_nodes lines are nodes: first digit is id, next two are coords
        node_dict = {}
        for _ in range(n_nodes):
            row = next(reader)
            node_id, *coordinates = row[0].split()
            node_dict[node_id] = _Point2D(*coordinates)
        
        assert(next(reader)[0] == 'EOF')

    graph = Graph(node_dict)

    return graph


# --- unit testing ---
class Test_Class2DMethods(unittest.TestCase):

    def test_distance(self):
        a = _Point2D(0., 0.)  # mix floats and ints
        b = _Point2D(3, 4)
        negative = _Point2D(-3, -4)
        z = object()

        # check simple egyptian triangle with sides 3-4-5
        self.assertEqual(a.distance_to(b), 5)
        # inverse should be the same for points on euclidean plane
        self.assertEqual(b.distance_to(a), 5)
        # distance to self should be zero
        self.assertEqual(a.distance_to(a), 0)
        # negative values should not matter
        self.assertEqual(a.distance_to(negative), 5)
        self.assertEqual(negative.distance_to(a), 5)

        # check that distance_to() fails when other is not a _Point2D
        with self.assertRaises(TypeError):
            a.distance_to(z)

    def test_slots(self):
        a = _Point2D(0., 0.)  # mix floats and ints
        # check __slots__ : initialization of new attribute fails
        with self.assertRaises(AttributeError):
            a.new_attribute = 'hello'

if __name__ == '__main__':
    unittest.main()
