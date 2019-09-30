"""
Crossover and mutation operators are taken from following study:

    Larranaga, P., Kuijpers, C. M. H., Murga, R. H., Inza, I., 
    & Dizdarevic, S. (1999). Genetic algorithms for the travelling 
    salesman problem: A review of representations and operators. 
    Artificial Intelligence Review, 13(2), 129-170.
"""

import heapq
from random import random, sample, shuffle


class Population:
    """ADT to represent population of solutions for genetic algorithm.

    Solution is a sequence of node ids constituting a tour.
    """

    # helper classes
    class _Item:
        """Lightweight ADT to represent a solution in our population.

        We use it to package information about fitness and tour together 
        to simplify management of heap. 
        Comparison is based on _fitness_val values."""

        __slots__ = '_fitness_val', '_tour'

        def __init__(self, fitness, tour):
            self._fitness_val = fitness
            self._tour = tour

        def __lt__(self, other):
            return self._fitness_val < other._fitness_val

        def __eq__(self, other):
            return self._fitness_val == other._fitness_val

    # class methods
    def __init__(self, graph):
        self._graph = graph
        self._item_heap = []  # min-oriented heap

    def initialize(self, size):
        """Randomly initializes population to specified size."""

        self._item_heap = []

        for _ in range(size):

            # construct possible tour, randomize order of nodes
            candidate_tour = list(self._graph.nodes())
            shuffle(candidate_tour)

            # package into _Item
            fitness = self._fitness(candidate_tour)
            item = self._Item(fitness, candidate_tour)

            # append to container
            self._item_heap.append(item)

        # transform container into min-oriented heap
        # makes smallest fitness lookup O(1), updating O(log n)
        heapq.heapify(self._item_heap)

        # postcondition
        assert len(self._item_heap) == size

    def _fitness(self, tour):
        """Returns fitness of a tour.

        smaller tour, larger is its fitness.
        """

        # make tour length an argument
        length = self._graph.tour_length(tour)

        return -1 * length

    def select_tour(self):
        """Selects a solution tour using 2-way Tournament Selection.

        Works in O(1).

        Taken from:
            Blickle, T., & Thiele, L. (1996). A comparison of selection 
            schemes used in evolutionary algorithms. Evolutionary 
            Computation, 4(4), 361-394.

        Returns:
            tour: list of nodes constituting a tour.
        """

        if not self._item_heap:
            raise Exception('Heap has not been initalized.')

        # choose two individuals randomly from population
        # without replacement
        chosen_items = sample(self._item_heap, 2)

        # select best-fitted item, compare on fitness
        best_item = max(chosen_items)

        return best_item._tour

    def update(self, tour):
        """Updates population with a tour if possible.

        If fitness of provided tour is higher than that of lowest-scoring 
        solution in population:
            1. Deletes lowest-scoring solution.
            2. Adds provided tour.
        Otherwise, nothing happens. Works in O(log n), where n is population 
        size.

        Args:
            tour: list of nodes constituting a tour.
        """

        if not self._item_heap:
            raise Exception('Heap has not been initalized.')

        new_item = self._Item(self._fitness(tour), tour)

        # our heap is min-oriented
        worst_item = self._item_heap[0]

        # whether new tour is better than worst one in population
        if new_item > worst_item:
            # remove lowest-scoring item, then add new item
            heapq.heapreplace(self._item_heap, new_item)

    def best_tour(self):
        """Returns best-fitted solution tour from population.

        Returns:
            tour: list of nodes constituting best tour.
        """

        if not self._item_heap:
            raise Exception('Heap has not been initalized.')

        # unpack an item from result list
        item, = heapq.nlargest(1, self._item_heap)

        return item._tour

    def is_saturated(self):
        """Returns True if all members have same fitness, False otherwise.

        Works in O(n), where n is population size.
        """

        if not self._item_heap:
            raise Exception('Heap has not been initalized.')

        # if all leaves of heap have same fitness as min item, then
        # all internal nodes have same fitness
        min_item = self._item_heap[0]

        # leaves of tree in heap array start at [n // 2]
        leaf_start = len(self._item_heap) // 2

        # check whether leaves have same values as minimum item
        return all(
            min_item == leaf
            for leaf in self._item_heap[leaf_start:]
        )


# main algorithm
def genetic(graph, population_size=200,
            max_iterations=50000, mutation_proba=0.1):
    """Estimates shortest tour in a graph using genetic algorithm.

    Args:
        graph: initialized instance of Graph.
        population_size: int, controls how many solutions population contains.
        max_iterations: int.
        mutation_proba: float in [0, 1], controls mutation probability.

    Returns:
        tour: list of nodes constituting shortest estimated tour.
    """

    # generate initial population
    population = Population(graph)
    population.initialize(population_size)

    # stopping condition - MAX_ITERATIONS limit reached
    for iteration in range(max_iterations):

        # select parents from population
        parent1 = population.select_tour()
        parent2 = population.select_tour()

        child1 = ox1(parent1, parent2)  # crossover
        sim(child1) if random() <= mutation_proba else None  # mutation
        population.update(child1)  # replacing ancestors

        # same for second child, but flip order of parents
        child2 = ox1(parent2, parent1)
        sim(child2) if random() <= mutation_proba else None
        population.update(child2)

        # stopping condition - oversaturated
        if (iteration % 1000 == 0):
            print(
                'Current best tour length: ',
                graph.tour_length(population.best_tour())
            )

            if population.is_saturated():
                break

    best_tour = population.best_tour()

    return best_tour


# --- Crossover Operators ---
def ox1(main_seq, secondary_seq):
    """Returns a list - result of OX1 order crossover between sequences.

    Works in O(len), where len is length of main_seq. 
    See Larranaga et al. (1999) for detailed explanation.

    Args:
        main_seq, secondary_seq: python sequences.
            Should contain same items and be of same length.

    Returns:
        child: list - result of applying OX1 crossover.
    """

    # preconditions
    length = len(main_seq)
    assert length == len(secondary_seq), \
        'Sequences must be of same length.'
    assert set(main_seq) == set(secondary_seq), \
        'Sequences must contain same elements.'

    # initialize child and get random slice
    child = length * [None]
    swath = _random_slice(length)

    # copy subtour from main parent into a child
    child[swath] = main_seq[swath]

    # fill child's missing genes with values from secondary_seq
    _fill_missing_genes(swath, source=secondary_seq, target=child)

    return child


def _fill_missing_genes(prefilled_slice, source, target):
    """Uses source's remaining alleles to fill missing genes in target."""

    source_length = len(source)
    start_point = prefilled_slice.stop

    # remember already filled values to skip them later
    already_filled_values = set(target[prefilled_slice])

    # keep target's offset to support simulatneous updates
    target_offset = 0

    # traverse source using offsetting, wrap around if needed
    for source_offset in range(source_length):
        idx_source = (source_offset + start_point) % source_length
        next_val = source[idx_source]

        if next_val not in already_filled_values:
            idx_target = (target_offset + start_point) % source_length

            # fill in target's gene with parent's allele
            target[idx_target] = next_val

            # move to next available place in target
            target_offset += 1


# --- Mutation Operators ---
def sim(seq):
    """Applies simple inversion mutator to a sequence.

    Modifies sequence in place by reversing its random portion. 
    See Larranaga et al. (1999) for detailed explanation.

    Args:
        seq: mutable sequence with length greater than one.    
    """

    # precondition -
    assert len(seq) > 0, \
        'Length of seq should be a positive integer'

    # select random portion of a sequence
    swath = _random_slice(len(seq))

    # reverse that portion
    seq[swath] = reversed(seq[swath])


# --- utility funcs ---
def _random_slice(seq_length):
    """Returns random slice object given sequence length.

    Args:
        seq_length (int): length of full sequence.
            Should be a positive integer.

    Returns:
        slice: start and end are chosen randomly.
            Contains at least one member.
    """

    # precondition
    assert isinstance(seq_length, int) and seq_length > 0, \
        'seq_length should be positive integer.'

    # +1 to consider last index of sequence
    cut_points = sample(range(seq_length + 1), 2)
    first_cut, last_cut = min(cut_points), max(cut_points)

    return slice(first_cut, last_cut)
