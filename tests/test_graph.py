import pytest

from src import graph


@pytest.fixture
def g(scope='module'):
    d = {
        'A': graph._Point2D(0, 3),
        'B': graph._Point2D(4, 0),
        'C': graph._Point2D(0, 0)
    }
    g = graph.Graph(d)


class Test_Point2D:

    def test_distance_to(self):
        a = graph._Point2D(0., 0)  # mix floats and ints
        b = graph._Point2D(3, 4)

        # check simple egyptian triangle with sides 3-4-5
        assert a.distance_to(b) == 5
        # inverse should be the same for points in euclidean space
        assert b.distance_to(a) == 5
        # distance to self should be inf
        assert a.distance_to(a) == float('inf')

        # negative values should not matter
        negative = graph._Point2D(-3, -4)
        assert a.distance_to(negative) == 5
        assert negative.distance_to(a) == 5

        # check that distance_to() fails when other is not a graph._Point2D
        z = object()
        with pytest.raises(TypeError):
            a.distance_to(z)


class TestGraph:

    def test_init(self):
        empty = {}
        with pytest.raises(ValueError):
            g = graph.Graph(empty)

        d = {'1': object(), '2': object()}
        with pytest.raises(TypeError):
            g = graph.Graph(d)
