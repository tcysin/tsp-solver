import math
import pytest

from src import greedy, graph


@pytest.fixture
def g(scope='module'):
    g = graph.read_csv('resources/a4_diamond.tsp')

    return g


#def test_greedy(g)