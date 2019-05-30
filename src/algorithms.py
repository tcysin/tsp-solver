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

from graph import Graph


# TODO:

def shortest_tour(graph, algorithm='genetic'):
    """Returns a list of nodes representing (approximate) shortest tour.
    
    Available algorithms: 'greedy', 'genetic', 'brute_force'.

    Arguments
    ---
        graph (Graph) -- isntance of a Graph
    """

    if isinstance(graph, Graph):

        if algorithm == 'genetic':
            # return _genetic(self)
            raise NotImplementedError()
        elif algorithm == 'greedy':
            return greedy(graph)
        else:
            raise ValueError('Passed algorithm is invalid.')

    else:
        raise TypeError('Provided graph is not an instance of a Graph.') 

# --- shortest tour finding algorithms ---
def genetic(graph):
    # TODO: implement genetic algorithm for shortest tour finding

    # population initialization

    # picking parents out of the population

    # crossover + mutation

    # updating the population

    # stopping condition

def greedy(graph):
    """Finds shortest tour using greedy approach."""

    tour = []
    # TODO: think about re-design of the whole thing with using
        # just tour list and going on until the list is full?
    # TODO: document why set
    available_nodes = set(graph.nodes())

    # pick random starting node in a graph, add it to tour path
    prev_node = available_nodes.pop()
    tour.append(prev_node)

    while available_nodes:
        # TODO: maybe get prev node from end of current tour list?
        # pick next node out of available nodes based on distance
        next_node = min(
            [candidate for candidate in available_nodes],
            key=lambda candidate: graph.distance(prev_node, candidate)
            )

        tour.append(next_node)
        available_nodes.remove(next_node)

        prev_node = next_node

    return tour