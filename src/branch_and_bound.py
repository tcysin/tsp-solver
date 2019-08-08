"""
Each branch in a tree represents a sub-tour. Each node represents 
addition of new vertex to parent's sub-tour. An addition decreases 
the number of available vertices. The leaves are last available 
vertices to add to the sub-tour. When leaves are appended, sub-tour 
becomes a tour.

We do not maintain space tree explicitly as a data structure. Instead, 
we construct it dynamically during exploration. 

The root of the tree is constructed manually.
"""

# TODO: make sure id interactions are with integers; now - with strs


from copy import deepcopy

import numpy as np

from .greedy import greedy


class SearchTree:

    class _Node:
        __slots__ = ['current', 'M', 'path', 'explored', 'bound']

        def __init__(self, current, M, path, explored, bound):
            self.current = current
            self.M = M
            self.path = path
            self.explored = explored
            self.bound = bound

    # class methods
    def __init__(self, graph):
        self._graph = graph
        self._tour_best = greedy(graph)
        self._length_best = graph.tour_length(self._tour_best)

        self._initialize_root()

    def _initialize_root(self):
        """Manually constructs root node in space search tree."""

        # data needed to assemble the root node
        start_node = next(self._graph.nodes())
        path = [start_node]
        explored_nodes = {start_node}
        # TODO: get graph's adjacency matrix
        M = 1
        M = np.array(M)

        # compute the bound of starting node
        reduction_cost = self._reduce_and_get_cost(M)  # reduction cost
        # total bound: weight from reduced + bound + prev_bound
        # there is no weight and no prev_bound
        bound = 0 + reduction_cost + 0

        # assemble the root
        self._root = self._Node(start_node, M, path, explored_nodes, bound)

    def _reduce_and_get_cost(self, M):
        """Reduces M in-place and returns cost of reduction."""

        # extract a vector of minimum values across the rows
        min_rows = M.min(axis=1)

        # gracefully handle infinity; 0 does not contribute to cost
        # and helps escape nan when subtracting inf from inf
        min_rows[min_rows == float('Inf')] = 0
        M -= min_rows.reshape((-1, 1))  # subtract these values from rows
        cost += sum(min_rows)  # update reduction cost

        # same for columns
        min_cols = M.min(axis=0)
        min_cols[min_cols == float('Inf')] = 0
        M -= min_cols.reshape((1, -1))
        cost += sum(min_cols)

        return cost

    def explore(self):
        # recursively traverse the children of the root
        for kid in self._children(self._root):
            self._DFS(kid)

    def _children(self, node):
        """Returns a generator over nodes' children in a graph.

        Constructs kids dynamically. 
        Each kid gets its own local copy of parent's data to work with.

        Arguments:
            node -- an instance of Node class; will let children to copy
                some of its attributes
            graph -- an instance of a Graph class
        """

        # get all the possible destination vertices
        all_vertices = set(self._graph.nodes())
        unexplored = all_vertices - node.explored

        for vertex in unexplored:

            # --- assemble an instance of kid node ---
            # vertex id under exploration
            vertex_kid = vertex

            # get local copies of parent's (node) data
            M_kid = deepcopy(node.M)

            path_kid = deepcopy(node.path)
            # update the path with fresh vertex
            path_kid.append(vertex_kid)

            explored_kid = deepcopy(node.explored)
            # add fresh vertex to explored set
            explored_kid.add(vertex_kid)

            bound_kid = node.bound

            # assemble an instance of a Node class
            node_kid = self._Node(vertex_kid,
                                  M_kid,
                                  path_kid,
                                  explored_kid,
                                  bound_kid)

            yield node_kid

    def _DFS(self, node):
        # stopping condition 1: end of tour is reached
        contains_valid_tour = len(node.path) == self._graph.size()
        if contains_valid_tour:
            self._update_best_solution(node.path)
            return

        # reduce the node, update its bound
        self._reduce(node)

        # stopping condition 2: bound violated
        if node.bound >= self._length_best:
            return

        else:
            # recursively explore the children of a node
            for kid in self._children(node):
                self._DFS(kid)

    def _update_best_solution(self, tour):
        """Updates best solution if passed tour is better."""

        assert len(tour) == len(self._tour_best), \
            'Argument is not a valid tour.'

        length = self._graph.tour_length(tour)
        if length < self._length_best:
            self._tour_best = tour
            self._length_best = length

    def _reduce(self, node):
        """Executes BnB edge inclusion and reduction over node matrix.

        Includes last edge in the node's path, reduces node's  matrix
        and updates the bound. Every operation is done in-place and
        modifies a node.
        """

        out_vertex, in_vertex = node.path[-2], node.path[-1]
        start = node.path[0]

        # remember the weight of an edge we are about to include
        weight = node.M[out_vertex][in_vertex]

        # include an edge by modifying node's matrix
        self._include_edge(out_vertex, in_vertex, start, node.M)

        # reduce the matrix, get the cost of the reductions
        cost = self._reduce_and_get_cost(node.M)

        # new bound = parents bound + edge weight + current reduction cost
        node.bound += weight + cost

    def _include_edge(self,
                      out_vertex,
                      in_vertex,
                      start,
                      M):
        """Modifies a matrix to include an edge.

        Sets a row, a column and a particular cell to infinity.
        This way, it abstracts edge inclusion from BnB algorithm."""

        # restrict all available edges incident to out_vertex
        M[out_vertex, :] = float('Inf')

        # restrict all the edges leading to in_vertex
        M[:, in_vertex] = float('Inf')

        # restrict the edge from in_vertex back to out_vertex
        M[out_vertex, in_vertex] = float('Inf')

        # restrict edge from out_vertex back to beginning of a tour
        M[in_vertex, start] = float('Inf')

        return


def branch_and_bound(graph):

    st = SearchTree(graph)
    st.explore()

    tour = st.best_tour()

    return tour