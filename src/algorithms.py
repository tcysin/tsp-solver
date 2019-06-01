"""
NAME
    algorithms

DESCRIPTION
    This module contains algorithms that work on graphs. Currently 
    supports algorithms for finding shortest tours.

FUNCTIONS
    shortest_tour(graph, algorithm='genetic') -> list

        Returns a sequence of nodes corresponding to the shortest tour 
        in the graph. List of supported algorithms:

            'greedy'
            'genetic'
            'brute_force'

Lower bound + tight lower bound on shortest tour
"""

from src.graph import Graph


# TODO:

def shortest_tour(graph, algorithm='genetic'):
    """Returns a list of nodes representing (approximate) shortest tour.

    Available algorithms: 'brute_force', 'genetic', 'greedy'.

    Params
    ---
        graph (Graph) -- isntance of Graph class
    """

    if not isinstance(graph, Graph):
        raise TypeError('Provided object is not a Graph.')

    if algorithm == 'brute_force':
        # return _brute_force
        raise NotImplementedError()
    if algorithm == 'genetic':
        # return _genetic(self)
        raise NotImplementedError()
    elif algorithm == 'greedy':
        return _greedy(graph)
    else:
        raise ValueError('Passed algorithm is invalid.')


# --- shortest tour finding algorithms ---


def genetic(graph):
    # TODO: implement genetic algorithm for shortest tour finding

    # population initialization

    # picking parents out of the population

    # crossover + mutation

    # updating the population

    # stopping condition
    pass


def _greedy(graph):
    """Finds shortest tour using greedy approach.

    O(n^2).
    """

    tour = []
    available_nodes = set(graph.nodes())

    # pick random starting node in a graph, add it to tour path
    prev_node = available_nodes.pop()
    tour.append(prev_node)

    while available_nodes:

        # pick next node out of available nodes based on distance
        next_node = min(
            (candidate for candidate in available_nodes),
            key=lambda x: graph.distance(prev_node, x)
        )

        tour.append(next_node)
        available_nodes.remove(next_node)

        prev_node = next_node

    return tour
