"""
DESCRIPTION
    This module provides graph algorithms.
"""

from .graph import read_csv
# algorithms
from .branch_and_bound import branch_and_bound
from .brute_force import brute_force
from .dynamic import dynamic
from .genetic import genetic
from .greedy import greedy

table = {
    'branch_and_bound': branch_and_bound,
    'brute_force': brute_force,
    'dynamic': dynamic,
    'genetic': genetic,
    'greedy': greedy
}

# TODO: re-design algo funcs to be classes accepting params
class Solver:
    """Solves instances of TSP problem.

    Available algorithms:
        'branch_and_bound', 'brute_force', 'dynamic', 'genetic', 'greedy'.
    """

    def __init__(self, algorithm='dynamic'):
        self.algorithm = table[algorithm]

    def solve(self, path):
        """Calculates and returns shortest tour in a graph.
        
        Returns:
            tour - list of nodes constituting shortest tour.
        """

        graph = read_csv(path)

        return self.algorithm(graph)
