import math
import os
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

    # positive sequence length
    # length of 1
    # length of 0
    # legative length

    with pytest.raises(AssertionError):
        genetic.random_slice(0)
    with pytest.raises(AssertionError):
        genetic.random_slice(-1)
