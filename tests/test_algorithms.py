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