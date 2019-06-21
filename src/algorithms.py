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
    prev_node = available_nodes.pop()
    tour.append(prev_node)

    while available_nodes:

        # pick next closest node out of available ones
        next_node = min(
            (candidate for candidate in available_nodes),
            key=lambda x: graph.distance(prev_node, x)
        )

        tour.append(next_node)
        available_nodes.remove(next_node)

        prev_node = next_node

    return tour
