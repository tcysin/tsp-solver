import math
import pytest
import random

from src import graph
from src import genetic


@pytest.fixture
def g(scope='module'):
    d = {
        'A': graph._Point2D(0, 0),
        'B': graph._Point2D(0, 2),
        'C': graph._Point2D(2, 2),
        'D': graph._Point2D(2, 0)
    }
    g = graph.Graph(d)

    return g


@pytest.fixture
def p(scope='module'):
    p = [
        (-4, ['A', 'B', 'C', 'D']),
        (-2, ['C', 'B', 'A', 'D']),
        (-7, ['D', 'C', 'A', 'B'])
    ]

    return p


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
    random.seed(7)

    seq = [1, 2, 3, 4, 5]
    genetic.sim(seq)
    assert seq == [3, 2, 1, 4, 5]

    with pytest.raises(AssertionError):
        genetic.sim([])
    with pytest.raises(AssertionError):
        genetic.sim([1])
    with pytest.raises(AssertionError):
        genetic.sim([1, 2])


def test_fill_missing_genes():
    source = [1, 2, 3, 4, 5]
    s = slice(2, 4)
    target = [None, None, 5, 2, None]

    genetic._fill_missing_genes(s, source, target)

    assert target == [3, 4, 5, 2, 1]


def test_ox1():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = list(reversed(a))
    result = genetic.ox1(a, b)

    assert len(result) == len(a)
    assert set(result) == set(a)


# TODO: check for comparison - longer tour has less fitness than smaller one
def test_get_fitness(g):
    good_tour = ['A', 'B', 'C', 'D']
    bad_tour = ['A', 'C', 'B', 'D']

    good_fitness = genetic._get_fitness(good_tour, g)
    # length is 2 + 2 + 2 + 2 == 8, fitness should be -8
    assert good_fitness == -8

    # optimal tour should have higher fitness value than bad tour
    bad_fitness = genetic._get_fitness(bad_tour, g)
    assert good_fitness > bad_fitness


def test_tournament_selection(p):

    random.seed(7)
    item = genetic.tournament_selection(p)
    assert item == (-2, ['C', 'B', 'A', 'D'])

    random.seed(2)
    item = genetic.tournament_selection(p)
    assert item == (-4, ['A', 'B', 'C', 'D'])


def test_select_two_parents(p):
    random.seed(2)
    p1, p2 = genetic.select_two_parents(p)
    assert p1 == ['A', 'B', 'C', 'D']
    assert p2 == ['C', 'B', 'A', 'D']


# TODO: test generate_population
# TODO: test select_parent
