"""
Implementation of Brute Force algorithm.

Checks all possible tours and selects the shortest one.
"""

from itertools import permutations


def brute_force(graph):
    """Calculates and returns shortest tour using brute force approach.

    Runs in O(n!). Provides exact solution.

    Args:
        graph: instance of a Graph.

    Returns:
        list: sequence of nodes constituting shortest tour.
    """

    all_tours = permutations(graph.nodes())
    best_length = float('inf')
    best_tour = None

    for tour in all_tours:
        # aggregate tour length in a smart way
        length = 0

        for src, dest in graph._edges_from_tour(tour):
            length += graph.distance(src, dest)

            # if length already bigger than best solution so far, stop
            if length >= best_length:
                length = float('inf')
                break

        # we found better tour - update variables
        if length < best_length:
            best_length = length
            best_tour = tour
        else:
            continue

    return list(best_tour)
