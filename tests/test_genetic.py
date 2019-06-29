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


def test_generate_swath():
    swath = genetic.generate_swath(10)

    assert isinstance(swath, slice)
    assert swath.start < swath.stop


    # positive sequence length
    # length of 1
    # length of 0
    # legative length

