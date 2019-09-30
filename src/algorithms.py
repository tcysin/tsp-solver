"""
DESCRIPTION
    This module provides graph algorithms.
"""

from .graph import read_csv
# algorithms
from .branch_and_bound import branch_and_bound
from .brute_force import brute_force
# from .dynamic import dynamic
from .genetic import genetic
from .greedy import greedy

table = {
    'branch_and_bound': branch_and_bound,
    'brute_force': brute_force,
    #'dynamic': None,
    'genetic': genetic,
    'greedy': greedy
}

# TODO: re-design algo funcs to be classes accepting params
class Solver:
    """Solves instances of TSP problem.

    Available algorithms:
        branch_and_bound
        brute_force
        dynamic
        genetic
        greedy
    """

    def __init__(self, algorithm='dynamic'):
        self.func = table[algorithm]

    def solve(self, path):
        graph = read_csv(path)

        return self.func(graph)