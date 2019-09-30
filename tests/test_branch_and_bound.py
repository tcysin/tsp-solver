import numpy as np
import pytest

from src.branch_and_bound import SearchTree
from src.graph import Graph


@pytest.fixture
def graph(scope='module'):
    # TODO: maybe randomness comes from here?
    d = {
        'A': graph._Point2D(0, 0),
        'B': graph._Point2D(0, 3),
        'C': graph._Point2D(4, 3),
        'D': graph._Point2D(4, 0)
    }
    g = Graph(d)

    return g

class TestSearchSpace:
    def test_reduce_and_get_cost(self):
        dummy = object()
        st = SearchTree(dummy)
        M = np.array([
            [1, 2],
            [19, 23]
        ])

        cost = st._reduce_and_get_cost(M)
        assert cost == 21

        result = np.array([[0,0], [0,3]])
        assert np.array_equal(M, result)

    def funcname(self, parameter_list):
        pass

