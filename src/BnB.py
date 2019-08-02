"""
In TSP, the main concept is that of a search space. We can think of it
as a tree (root as starting vertex). BnB searching boils down to
traversing this tree in some intelligent manner.

The key is to understand what a node represents. Node tells something 
about how our search is
going: how many edges are already explored, matrices constructed,
etc. It encodes a state of a search space.

Nodes need their own variables and might retain some information from
the parent.

In case of simple BnB, a node represents a vertex ready to be added to
previously explored sub-path.
It needs to retain some information from a parent: parent's adjacency
matrix, parent's bound and sub-path.
It will have some things of its own: its own adjacency matrix, its
own bound and a new sub-path (just current node added to previous
sub-path).
The node will pass on some information to its kids, which are nodes
themselves.
"""

import numpy as np

from copy import deepcopy
from time import clock


from graph import Graph
from greedy import greedy


class Node:
    """Lightweight container for a tree node.

    Attibutes:
        current (int) -- id of a current vertex
        path (list) -- a sub-path of a tree leading to current vertex
        M -- a matrix corresponding to current position in search tree
        explored (set) -- a set of already explored vertices
        bound (float) -- a bound of a current node 
        
    """
    __slots__ = ['current', 'M', 'path', 'explored', 'bound']
    
    def __init__(self, current, M, path, explored, bound):
        self.current = current
        self.M = M
        self.path = path
        self.explored = explored
        self.bound = bound


# --- helper functions ---
def contains_valid_tour(node, tree):
    """Returns True if node's path is a valid tour, False otherwise."""
    return len(node.path) == tree.size()

def get_children(node, tree):
    """Returns a generator over nodes' children in a tree.

    Each kid gets its own local copy of parent's data to work with.

    Arguments:
        node -- an instance of Node class; will let children to copy
            some of its attributes
        tree -- an instance of a Graph class
    """

    all_vertices = set(tree.vertices())

    # get all the possible destination vertices
    for vertex in get_unexplored_vertices(node.explored,
         all_vertices):

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
        node_kid = Node(vertex_kid,
                        M_kid,
                        path_kid,
                        explored_kid,
                        bound_kid)
        
        yield node_kid

def get_unexplored_vertices(explored, all_vertices):
    """Returns a set of unexplored vertices.

    Arguments:
        explored (set) -- set of vertices explored so far
        all_vertices (set) -- set of all vertices
    """
    return all_vertices - explored

def is_bound_worse(node):
    """Returns True if node's bound is bigger than global best solution."""
    global length_best
    return node.bound >= length_best

def include_edge(beginning_node,
                 end_node,
                 starting_point,
                 M):
    """Modifies a matrix to include an edge.

    Sets some a row, a column and a particular cell to infinity.
    This way, it abstracts edge inclusion from BnB algorithm."""

    # restrict all available edges incident to beginning_node
    M[beginning_node, :] = float('Inf')

    # restrict all the edges leading to end_node
    M[:, end_node] = float('Inf')

    # restrict the edge from end_node back to beginning_node
    M[beginning_node, end_node] = float('Inf')

    # restrict edge from beginning_node back to the very beginning of a tour
    M[end_node, starting_point] = float('Inf')

    return
    
def process(node):
    """Executes BnB edge inclusion and reduction over matrix.

    Includes last edge in the node's path, reduces node's  matrix
    and updates the bound. Every operation is done in-place and
    modifies a node.
    """

    out_node, in_node = node.path[-2], node.current
    start = node.path[0]

    # remember the weight of an edge we are about to include
    weight = node.M[out_node][in_node]

    # include an edge by modifying node's matrix
    include_edge(out_node, in_node, start, node.M)

    # reduce the matrix, get the cost of the reductions
    cost = reduce_and_get_cost(node.M)
    
    # new bound - parents bound + edge weight + current reduction cost
    node.bound += weight + cost

    return

def reduce_and_get_cost(M):
    """Reduces M in-place and returns cost of reduction."""

    # extract a vector of minimum values across the rows
    min_rows = M.min(axis=1)
    
    # gracefully handle infinity; 0 does not contribute to cost
    # and helps escape nan when subtracting inf from inf
    min_rows[min_rows == float('Inf')] = 0
    # subtract these values from rows
    M -= min_rows.reshape((-1,1))

    # vector of minum values across the columns
    min_cols = M.min(axis=0)
    
    # gracefully handle infinity
    min_cols[min_cols == float('Inf')] = 0
    M -= min_cols.reshape((1,-1))

    # calculate the reduction cost
    cost = sum(min_rows) +  sum(min_cols)

    return cost

def update_best_solution(node, tree):
    """Updates best global tour and length.

    If tour ending at node is better than global tour, replace best
    tour length and actual path with node's own.
    """
    global length_best
    global tour_best

    length_current = tree.get_tour_length(node.path)
        
    if length_best > length_current:
        length_best = length_current
        tour_best = node.path

    return


# --- tree traversal logic ---
def DFS(node, tree):
    """Emulates depth-first, lazy search over a tree."""

    # stopping condition 1: end of tour is reached
    if contains_valid_tour(node, tree):
        update_best_solution(node, tree)
        return
    
    # do something with the node using its parent information
    process(node)
    
    # check the stopping condition
    if is_bound_worse(node):
        return

    # recursively explore the children of a node
    children = get_children(node, tree)
    
    for kid in children:
        DFS(kid, tree)


# --- Branch and Bound algorithm --- 
def BnB(graph):

    print('Starting BnB...')
    start = clock()
    
    # --- initial step ---
    global tour_best
    tour_best = greedy(graph)
    global length_best
    length_best = graph.get_tour_length(tour_best)
    


    # --- construct root of the tree ---
    # data needed to assemble a root
    starting_vertex = next(graph.vertices())
    path = [starting_vertex]
    explored_nodes = { starting_vertex }
    M = graph.get_adjacency_matrix()
    # convert matrix to numpy representation
    M = np.array(M)


    # --- process first node manually ---
    # get the bound, reduce a matrix in-place
    reduction_cost = reduce_and_get_cost(M)

    # total bound: weight from reduced + bound + prev_bound
    # there is no weight and no prev_bound
    bound = 0 + reduction_cost + 0

    # --- explore the kids of the root node ---
    # assemble the root
    root = Node(starting_vertex,
                M,
                path,
                explored_nodes,
                bound=bound)

    # recursively explore children of a root
    children = get_children(node=root, tree=graph)
    
    for kid in children:
        DFS(kid, tree=graph)

    print('BnB finished. Elapsed time:', clock() - start)
    print('Best tour:', tour_best)
    print('Length', length_best)

    return tour_best
    

if __name__ == '__main__':
    vertices = {'a', 'b', 'c', 'd'}
    initial_path = ['a']

    #print(list(available_kids(path, vertices)))