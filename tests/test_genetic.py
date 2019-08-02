# TODO: design tests for population class

import math
import pytest
import random

from src import graph
from src import genetic


@pytest.fixture
def g(scope='module'):
    d = {
        'A': graph._Point2D(0, 0),
        'B': graph._Point2D(0, 3),
        'C': graph._Point2D(4, 3),
        'D': graph._Point2D(4, 0)
    }
    g = graph.Graph(d)

    return g

# TODO: think whether you really need fixture for Population
@pytest.fixture
def p(g, scope='module'):
    p = genetic.Population(g)

    return p


class Test_Population:

    class Test_Item:

        def test_init(self):
            good_tour_item = genetic.Population._Item(-13, [1, 2, 3, 4])
            bad_tour_item = genetic.Population._Item(-20, [1, 3, 4, 2])

            assert bad_tour_item < good_tour_item
            assert not (bad_tour_item > good_tour_item)
            assert good_tour_item == good_tour_item

            # checking __slots__
            with pytest.raises(AttributeError):
                good_tour_item.hello = 15

    def test_initialize(self, g):
        #p = genetic.Population(g)
        pass

    def test_fitness(self, p):
        tour1 = ['A', 'B', 'C', 'D']
        tour2 = ['B', 'C', 'D', 'A']
        assert p._fitness(tour1) == -14
        assert p._fitness(tour2) == -14

    def test_select_tour(self, p):
        pass

    def test_update(self, g):
        p = genetic.Population(g)
        pass

    def test_best_tour(self, g):

        p = genetic.Population(g)
        p.initialize(3)
        pass


def test_random_slice():
    s = genetic._random_slice(1)
    assert isinstance(s, slice)
    assert s.start == 0 and s.stop == 1

    with pytest.raises(AssertionError):
        genetic._random_slice(0)
    with pytest.raises(AssertionError):
        genetic._random_slice(-1)
    with pytest.raises(AssertionError):
        genetic._random_slice('a')


def test_sim():
    random.seed(3)

    seq = [1, 2, 3, 4, 5]
    genetic.sim(seq)
    assert seq == [1, 4, 3, 2, 5]

    with pytest.raises(AssertionError):
        genetic.sim([])


def test_fill_missing_genes():
    source = [1, 2, 3, 4, 5]
    s = slice(2, 4)
    target = [None, None, 5, 2, None]

    genetic._fill_missing_genes(s, source, target)

    assert target == [3, 4, 5, 2, 1]


def test_ox1():
    random.seed(7)
    a = [1, 2, 3, 4, 5]
    b = list(reversed(a))
    result = genetic.ox1(a, b)

    assert len(result) == len(a)
    assert set(result) == set(a)
    assert result == [4,2,3,1,5]
