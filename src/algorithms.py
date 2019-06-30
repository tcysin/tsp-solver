"""
DESCRIPTION
    This module provides graph algorithms.
    
    Algorithms available:
        - greedy
        - genetic algorithm
"""


def greedy(graph):
    """Finds shortest tour using greedy approach.

    Runs in O(n^2). Provides approximate solution.

    Args:
        graph (Graph): instance of a Graph.

    Returns:
        list: sequence of nodes constituting shortest tour.

    """

    tour = []
    available_nodes = set(graph.nodes())

    # pick random starting node, add it to tour path
    starting_node = available_nodes.pop()
    tour.append(starting_node)

    while available_nodes:
        prev_node = tour[-1]

        # pick next closest node out of available ones
        next_node = min(
            (candidate for candidate in available_nodes),
            key=lambda x: graph.distance(prev_node, x)
        )

        tour.append(next_node)
        available_nodes.remove(next_node)

    return tour


# utility functions
def tour_length(tour, graph):
    """Returns the length of the tour in a graph."""

    assert set(tour) == set(graph.nodes())

    length = 0

    for src, dest in edges_from_tour(tour):
        length += graph.distance(src, dest)

    return length
    

def edges_from_tour(tour):
    """Returns iterator over edges in a tour. Includes edge from end to start."""

    for edge in zip(tour, tour[1:]):
        yield edge

    # yield last edge from end to start
    circling_edge = tour[-1], tour[0]
    yield circling_edge