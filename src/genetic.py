"""

Crossover and mutation operators are taken from the following study:

    Larranaga, P., Kuijpers, C. M. H., Murga, R. H., Inza, I., 
    & Dizdarevic, S. (1999). Genetic algorithms for the travelling 
    salesman problem: A review of representations and operators. 
    Artificial Intelligence Review, 13(2), 129-170.
"""

import heapq
from operator import itemgetter
from random import randint, random, sample
from statistics import mean

from .algorithms import greedy, tour_length


# taken from study above
POPULATION_SIZE = 200

MAX_ITERATIONS = 50000
ITERATIONS_WITHOUT_IMPROVEMENT = 1000

CROSSOVER_PROBA = 1.0
MUTATION_PROBA = 0.1


def genetic(graph):

    # generate initial population
    population = generate_population(graph, POPULATION_SIZE)

    # get initial fitness of the population
    #prev_mean_fitness = mean_fitness(population)

    #no_improvement_counter = 0

    # stopping condition - MAX_ITERATIONS limit reached
    for _ in range(MAX_ITERATIONS):

        # make sure population is not overly saturated
        # if is_population_saturated(population):
        #    print('Population oversaturated. Returning best solution.')
        #    break

        # select parents from the population
        parent1, parent2 = select_two_parents(population)

        # apply OX1 crossover w. high probability proba_crossover
        if random() <= CROSSOVER_PROBA:
            child1 = ox1(parent1, parent2)
        if random() <= CROSSOVER_PROBA:
            child2 = ox1(parent2, parent1)

        # apply SIM mutation w. low proba_mutation
        if random() <= MUTATION_PROBA:
            sim(child1)
        if random() <= MUTATION_PROBA:
            sim(child2)

        # replace ancestors with children in a population
        child1_item = _get_fitness(child1, graph), child1
        update_population(child1_item, population)

        child2_item = _get_fitness(child2, graph), child2
        update_population(child2_item, population)

        # check for no improvement
        #current_mean_fitness = mean_fitness(population)
        #is_fitness_improved = current_mean_fitness > prev_mean_fitness

        # reassign prev_average_fitness to be current value
        #prev_mean_fitness = current_mean_fitness

        # if not is_fitness_improved:
        #    no_improvement_counter += 1
        # else:
        # in case of improvement, reset the counter
        #    no_improvement_counter = 0

        # if no_improvement_counter >= ITERATIONS_WITHOUT_IMPROVEMENT:
        #    s = ITERATIONS_WITHOUT_IMPROVEMENT
        # print('\nNo improvement recorded for {} iterations'.format(s))
        # print('Stopping Genetic Algorithm.')
        # break

    #best_solution = heapq.nlargest(1, population)[0][1]
    _, best_tour = max(population, key=itemgetter(0))

    return best_tour


# --- Helper functions ---
def mean_fitness(population):
    """Returns average fitness of population."""

    avg = mean(item[0] for item in population)
    return round(avg, 2)


def is_population_saturated(population):
    """Checks wheter all values in population are same."""
    return len(set(fitness for (fitness, _) in population)) <= 1


def update_population(item, population):
    """Updates population with new item if item is good enough.

    If provided items fitness is better than that of worst item in
    population, remove worst item and add provided item.
    """

    # if items fitness is better than population's worst items fitness
    if item[0] > population[0][0]:
        # get rid of worst item, add this one
        heapq.heappushpop(population, item)

    return


# --- Population Initialization ---
def generate_population(graph, size):
    """Returns initial population of possible solutions.

    Uses heuristic to approximate shortest tour and then repeatedly 
    mutates it to construct other members.

    Taken from:
        Abdoun, O., Abouchaka, J. (2011). A Comparative Study of Adaptive 
        Crossover Operators for GA to Resolve TSP.

    Returns:
        set: contains tuples (fitness score, tour)
    """

    population = set()

    # construct first individual, add it to population
    initial_tour = greedy(graph)
    item = _get_fitness(initial_tour, graph), initial_tour
    population.add(item)

    # mutate initial solution to generate remaining members
    for _ in range(size - 1):
        mutated_tour = initial_tour[:]
        sim(mutated_tour)

        item = _get_fitness(mutated_tour, graph), mutated_tour
        population.add(item)

    return population


def _get_fitness(tour, graph):
    """Returns fitness of a tour.

    The smaller the tour, the larger is its fitness.
    """

    length = tour_length(tour, graph)

    return -1 * length


def select_two_parents(population):
    """Selects two parent solutions from population."""

    parent1 = tournament_selection(population)
    parent2 = tournament_selection(population)  # could be same as parent1

    return parent1, parent2


def tournament_selection(population):
    """Returns an item from the population.
    
    Uses 2-way tournament selection to choose a solutions.

    Taken from:
        Blickle, T., & Thiele, L. (1996). A comparison of selection schemes 
        used in evolutionary algorithms. Evolutionary Computation, 4(4), 
        361-394.
    """

    # TODO
    # https://www.cse.unr.edu/~sushil/class/gas/papers/Select.pdf
    # https://pdfs.semanticscholar.org/fef8/1135f587851f19fe515cb8eb3812e3706b27.pdf
    

    # choose a number of individuals randomly from population
        # with or without replacement
    chosen_items = sample(population, 2)

    # select best-fitted individual from chosen ones
    best_item = max(chosen_items, key=itemgetter(0))
    _, tour = best_item
    
    return tour

# --- Crossover Operators ---
def ox1(main_parent, secondary_parent):
    """Returns a result of OX1 order crossover between parents.

    Copies random subtour of main_parent into child. Then uses 
    secondary_parent to fill in missing genes, preserving relative 
    order of parent's genes.

    Example: 
        parent1 = [A,B,C,D,E]
        parent2 = [E,D,C,B,A]
        randomly chosen subtour from parent1 = [B,C]
        child before filling: [_, B, C, _, _]
        starting position at parent2: [E,D,C, -> B ,A]
        child after filling: [D, B, C, A, E]
    """

    # if parents happen to be the same
    if main_parent == secondary_parent:
        return main_parent

    # preconditions
    length = len(main_parent)
    assert(length == len(secondary_parent))

    swath = random_slice(length)
    # TODO: maybe move this quality assurance routine somewhere?
    # check if the swath contains only 1 member
    while swath.stop - swath.start <= 1:
        swath = random_slice(length)
    # make sure the swath does not cover the whole parent
    while swath.start == 0 and swath.stop == length:
        swath = random_slice(length)

    child = length * [None]

    # copy subtour from main parent into a child
    child[swath] = main_parent[swath]

    # fill child's missing genes with alleles from secondary_parent
    fill_missing_genes(swath, source=secondary_parent, target=child)

    return child


def fill_missing_genes(prefilled_slice, source, target):
    """Uses source's remaining alleles to fill missing genes in target."""

    source_length = len(source)
    start_point = prefilled_slice.stop

    # remember already filled values to skip them later
    redundant_values = set(target[prefilled_slice])

    # keep target's offset to support simulatneous updates
    target_offset = 0

    # traverse the source using offsetting
    for source_offset in range(source_length):
        # starting at specified point of source
        # wrapping around if needed
        idx_source = (source_offset + start_point) % source_length

        # pick next allele of interest from source
        value = source[idx_source]

        if value not in redundant_values:
            # wrap around
            idx_target = ((target_offset + start_point)
                          % source_length)

            # fill in target's gene with parent's allele
            target[idx_target] = value

            # move to next available place in target
            target_offset += 1

    return


# --- Mutation Operators ---
def sim(seq):
    """Applies simple inversion mutator to a sequence.

    Modifies seq in place by reversing random sub-sequence. Length of 
    random sub-sequence is guranteed to be at least two.

    Args:
        seq: sequence with length greater than one.

    Example: sim([1, 2, 3, 4, 5]) -> [3, 2, 1, 4, 5]
    """

    # precondition
    assert len(seq) > 1, \
        'length of seq should be greater than 2.'

    # select random portion of a sequence
    swath = random_slice(len(seq))

    # check if the swath contains only 1 member
    while swath.stop - swath.start <= 1:
        swath = random_slice(len(seq))

    # reverse that portion
    seq[swath] = reversed(seq[swath])


def random_slice(seq_length):
    """Returns random slice object given sequence length.

    Args:
        seq_length (int): length of full sequence.
            Should be positive.

    """

    # precondition
    assert isinstance(seq_length, int) and seq_length > 0, \
        'seq_length should be positive integer.'

    # +1 to consider last index of sequence as well
    cut_points = sample(range(seq_length + 1), 2)
    first_cut, last_cut = min(cut_points), max(cut_points)

    return slice(first_cut, last_cut)
