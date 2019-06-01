"""
Larranaga, P., Kuijpers, C. M. H., Murga, 
R. H., Inza, I., & Dizdarevic, S. (1999). 

Genetic algorithms for the travelling salesman problem: 
A review of representations and operators. 

Artificial Intelligence Review, 13(2), 129-170.
"""

import heapq
from random import randint, random, sample
from statistics import mean

from greedy import greedy


# taken from study above
POPULATION_SIZE = 200

MAX_ITERATIONS = 50000
ITERATIONS_WITHOUT_IMPROVEMENT = 1000

CROSSOVER_PROBA = 1.0
MUTATION_PROBA = 0.1


# ----- Helper functions -----
def mutate(chromosome, func):
    """Applies provided mutation func to chromosome."""

    return func(chromosome)


def crossover(main_parent, secondary_parent, func):
    """Applies provided crossover func to main_parent and secondary_parent."""

    return func(main_parent, secondary_parent)


def generate_swath(sequence_length):
    """Generates random slice for sequence of given length."""

    cut_points = sample(range(sequence_length), 2)
    first_cut, last_cut = min(cut_points), max(cut_points)

    return slice(first_cut, last_cut)


def get_fitness(chromosome, graph):
    """Returns fitness of a chromosome. Tour length in our case."""

    return round(abs(graph.get_tour_length(chromosome) - graph.upper_bound()))


def mean_fitness(population):
    """Returns average fitness of population."""

    avg = mean(item[0] for item in population)
    return round(avg, 2)


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


def select_parent(population):
    """Selects two chromosomes from population with Roulette Wheel selection.

    Potvin, J.-Y. (1996). Genetic algorithms for the TSP.
    """

    # sum up fitnesses over whole population
    total = sum(item[0] for item in population)

    # generate random number in range [0, total - 1)
    num = randint(0, total - 1)

    # sum up values of chromosome, select the first one which exceeds
    # chosen number
    running_sum = 0.
    for item in population:

        # TODO: the problem is somewhere here
        running_sum += item[0]

        # if running sum exceeds the n, then choose this chromosome
        # the probability is proportional to its fitness value
        if running_sum > num:
            return item[1]
    print('hitting overflow error')
    return population[0][1]


def select_two_parents(population):
    """Selects two different parent chromosomes from population."""

    parent1 = select_parent(population)
    parent2 = select_parent(population)

    """
    # prevent choice of same parent - loops here
    while parent1 == parent2:
        parent2 = select_parent(population)
    """

    return parent1, parent2


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


# ----- Population Initialization -----
# Abdoun, O., Abouchaka, J. (2011). A Comparative Study of
#   Adaptive Crossover Operators for GA to Resolve TSP.
def generate_population(graph):
    """Generates initial population of (fitness, possible solution) items.

    Uses heuristic to approximate shortest tour and then
    repeatedly mutates it to construct other members.

    Returns
    ---
        list: contains tuples (fitness score, tour)
    """

    population = []

    # use heuristics to construct first individual
    initial_chromosome = greedy(graph)

    # add (fitness, initial_chromosome) to population
    item = get_fitness(initial_chromosome, graph), initial_chromosome
    population.append(item)

    # mutate the initial chromosome to generate remaining members
    for _ in range(POPULATION_SIZE - 1):

        # copy to avoid changing initial one in-place
        mutated_chromosome = initial_chromosome[:]
        mutate(mutated_chromosome, sim)

        # add mutant to population
        item = get_fitness(mutated_chromosome, graph), mutated_chromosome
        population.append(item)

    # heapify the population
    heapq.heapify(population)

    return population


# ----- Crossover Operators -----
def ox1(main_parent, secondary_parent):
    """Returns a result of OX1 order crossover between parents.

    Randomly chooses a subtour of main_parent and copies it
    into a child. Then uses secondary_parent to fill in 
    missing genes, preserving relative order of parent's genes.

    Example: 
    parent1 = [A,B,C,D,E]
    parent2 = [E,D,C,B,A]
    randomly chosen subtour from parent1 = [B,C]
    child before filling: [_, B, C, _, _]
    child after filling: [D, B, C, A, E]
    """

    # if parents happen to be the same
    if main_parent == secondary_parent:
        return main_parent

    length = len(main_parent)
    assert(length == len(secondary_parent))

    # initialize an empty child
    child = length * [None]

    # choose random swath of genes between cut points
    swath = generate_swath(length)

    # check if the swath contains only 1 member
    while swath.stop - swath.start <= 1:
        swath = generate_swath(length)

    # make sure the swath does not cover the whole parent
    while swath.start == 0 and swath.stop == length:
        swath = generate_swath(length)

    # and copy it from main parent into a child at the same indices
    child[swath] = main_parent[swath]

    # fill child's missing genes with alleles from secondary_parent
    fill_missing_genes(swath, source=secondary_parent, target=child)

    return child

# ----- Mutation Operators -----


def sim(chromosome):
    """Apply simple inversion mutator to a chromosome.

    Modifies in-place by reversing random sub-sequence. 
    """

    # get random portion of a chromosome
    swath = generate_swath(len(chromosome))

    # check if the swath contains only 1 member
    while swath.stop - swath.start <= 1:
        swath = generate_swath(len(chromosome))

    # reverse that portion
    chromosome[swath] = reversed(chromosome[swath])

    return


def genetic(graph):

    # generate initial population
    population = generate_population(graph)

    # get initial fitness of the population
    prev_mean_fitness = mean_fitness(population)

    no_improvement_counter = 0

    # stopping condition - MAX_ITERATIONS limit reached
    for _ in range(MAX_ITERATIONS):

        # make sure population is not overly saturated
        if is_population_saturated(population):
            print('Population oversaturated. Returning best solution.')
            break

        # select parents from the population
        parent1, parent2 = select_two_parents(population)

        # apply OX1 crossover w. high probability proba_crossover
        if random() <= CROSSOVER_PROBA:
            child1 = crossover(parent1, parent2, ox1)
        if random() <= CROSSOVER_PROBA:
            child2 = crossover(parent2, parent1, ox1)

        # apply SIM mutation w. low proba_mutation
        if random() <= MUTATION_PROBA:
            mutate(child1, sim)
        if random() <= MUTATION_PROBA:
            mutate(child2, sim)

        # replace ancestors with children in a population
        child1_item = get_fitness(child1, graph), child1
        update_population(child1_item, population)

        child2_item = get_fitness(child2, graph), child2
        update_population(child2_item, population)

        # check for no improvement
        current_mean_fitness = mean_fitness(population)
        is_fitness_improved = current_mean_fitness > prev_mean_fitness

        # reassign prev_average_fitness to be current value
        prev_mean_fitness = current_mean_fitness

        if not is_fitness_improved:
            no_improvement_counter += 1
        else:
            # in case of improvement, reset the counter
            no_improvement_counter = 0

        if no_improvement_counter >= ITERATIONS_WITHOUT_IMPROVEMENT:
            s = ITERATIONS_WITHOUT_IMPROVEMENT
            print('\nNo improvement recorded for {} iterations'.format(s))
            print('Stopping Genetic Algorithm.')
            break

    best_solution = heapq.nlargest(1, population)[0][1]

    return best_solution
