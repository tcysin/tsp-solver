import math
import os
import pytest

from src import algorithms
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


def test_edges_from_tour():
    tour = [1, 2, 3, 4]

    assert (
        list(algorithms.edges_from_tour(tour))
        == [(1, 2), (2, 3), (3, 4), (4, 1)]
    )


def test_tour_length(g):
    tour = ['A', 'B', 'C']

    assert algorithms.tour_length(tour, g) == 3+4+5