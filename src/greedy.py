"""
Implementation of Nearest Neighbour (greedy) algorithm.

Start at random node and repeatedly visit nearest node until all are 
visited.
"""


def greedy(graph, **kwargs):
    """Calculates and returns shortest tour using greedy approach.

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
        # continue from previously added node
        prev_node = tour[-1]

        # pick next closest node out of available ones
        next_node = min(
            (candidate for candidate in available_nodes),
            key=lambda x: graph.distance(prev_node, x)
        )

        tour.append(next_node)
        available_nodes.remove(next_node)

    return tour
