import math
import os
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

    return g


class Test_Point2D:

    def test_init(self):
        with pytest.raises(TypeError):
            _ = graph._Point2D('one', 'two')

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
            _ = graph.Graph(empty)

        d = {'1': object(), '2': object()}
        with pytest.raises(TypeError):
            _ = graph.Graph(d)

    def test_nodes(self, g):
        assert set(g.nodes()) == set(['A', 'B', 'C'])

    def test_size(self, g):
        assert g.size() == 3

    def test_distance(self, g):
        assert g.distance('A', 'B') == 5
        assert g.distance('A', 'C') == 3
        assert g.distance('C', 'B') == 4

        # test distance to something not in a graph
        with pytest.raises(KeyError):
            g.distance('A', 'spam')

    def test_edges_from_tour(self, g):
        tour = [1, 2, 3, 4]

        assert (
            list(g._edges_from_tour(tour))
            == [(1, 2), (2, 3), (3, 4), (4, 1)]
        )

    def test_tour_length(self, g):
        tour = ['A', 'B', 'C']

        assert g.tour_length(tour) == 3+4+5


def test_read_csv():
    gr = graph.read_csv('tests/resources/a4_diamond.tsp')

    assert set(gr.nodes()) == set([0, 1, 2, 3])

    assert (
        gr.distance(0, 2)
        == gr.distance(2, 3)
        == gr.distance(3, 1)
        == gr.distance(1, 0)
        == math.sqrt(8)
    )

    assert (
        gr.distance(0, 3)
        == gr.distance(2, 1)
        == 4
    )

    # TODO: add exception cases
