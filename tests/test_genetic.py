import math
import pytest
import random

from src import graph
from src import genetic


@pytest.fixture
def g(scope='module'):
    d = {
        'A': graph._Point2D(0, 3),
        'B': graph._Point2D(4, 0),
        'C': graph._Point2D(0, 0)
    }
    g = graph.Graph(d)

    return g


def test_random_slice():
    s = genetic.random_slice(1)
    assert isinstance(s, slice)
    assert s.start == 0 and s.stop == 1

    with pytest.raises(AssertionError):
        genetic.random_slice(0)
    with pytest.raises(AssertionError):
        genetic.random_slice(-1)
    with pytest.raises(AssertionError):
        genetic.random_slice('a')


def test_sim():
    random.seed(7)

    seq = [1, 2, 3, 4, 5]
    child = genetic.sim(seq)
    assert child == [3, 2, 1, 4, 5]

    with pytest.raises(AssertionError):
        genetic.sim([])
    with pytest.raises(AssertionError):
        genetic.sim([1])
