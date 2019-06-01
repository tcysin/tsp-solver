import math
import pytest

from src import algorithms, graph


@pytest.fixture
def g(scope='module'):
    g = graph.read_csv('resources/a4_diamond.tsp')

    return g


def test_shortest_tour(g):
    with pytest.raises(TypeError):
        dummy = object()
        algorithms.shortest_tour(dummy)

    with pytest.raises(ValueError):
        algorithms.shortest_tour(g, 'random name of an algorithm')
