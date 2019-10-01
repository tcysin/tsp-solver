def dynamic(graph):
    """Returns shortest tour using dynamic programming approach.

    The idea is to store lengths of smaller sub-paths and re-use them
    to compute larger sub-paths.
    """

    adjacency_M = graph.adjacency_matrix()

    tour = _dynamic(adjacency_M, start_node=0)

    return tour


def _dynamic(M, start_node):
    """
    TODO: The state of search is encoded using memo table.
    The row index indicates ...
    The column index translated to binary 1011.. encodes the sub-path.

    Args:
        M - adjacency matrix of a graph.
        start_node - id of the beginning node.
    """

    # preconditions
    verify(M, start_node)

    # initialize memo table of shape (N, 2^N)
    n_nodes = len(M)
    memo = [
        [None] * (2**n_nodes)  # 2^N cols
        for _ in range(n_nodes)  # N rows
    ]

    # pre-fill the memo table
    setup(M, start_node, memo)

    # progressively build memo and find solution
    solve(M, start_node, memo)

    tour = find_optimal_tour(M, start_node, memo)

    return tour


def verify(M, start_node):
    """Make sure the adjacency matrix and start node are valid."""

    n_nodes = len(M)

    if n_nodes <= 2:
        raise Exception('Not enough nodes in the graph.')

    # square adj matrix check
    if not all(n_nodes == len(M[i]) for i in range(n_nodes)):
        raise Exception('Matrix must be square (n x n).')

    # start node out of range
    if (start_node < 0) or (n_nodes <= start_node):
        raise Exception('Invalid start node.')

    # too many nodes
    if n_nodes > 28:
        raise Exception(
            """Matrix too large! A matrix that size for the
            DP TSP problem with a time complexity of O(n^2*2^n)
            requires way too much computation for any modern home
            computer to handle."""
        )


def setup(M, start_node, memo):
    """Fills the memo table with solutions for sub-paths of length 2.

    We encode subpath with a binary representation of an integer.
    Indices of active bits correspond to nodes included in the subpath.
    For example, 0101 means that the nodes {0, 2} are included.
    """

    n_nodes = len(M)

    for node in range(n_nodes):

        if node == start_node:
            continue

        # store the length of an edge from each node to start node
        encoded_state = (1 << start_node) | (1 << node)  # i.e. '10001'
        memo[node][encoded_state] = M[node][start_node]


def solve(M, start, memo):
    """Finds optimal solutions for sub-paths of length 3 and greater.

    Builds memo table by saving and re-using solutions.
    """

    n_nodes = len(M)

    # solve for all subpaths at lengths [3, 4, 5 ... n_nodes]
    # remember and re-use solutions to earlier subpaths
    for len_subpath in range(3, n_nodes + 1):

        # solve for all subpaths of given length
        for subset in bit_combos(len_subpath, n_nodes):
            # bit_combos generates encoding of all subpaths of given length
            # subset represents the subset of visited nodes

            # enforce starting node to be in the subset
            # otherwise, we cannot eventually get to the starting node
            if notIn(start, subset):
                continue

            # candidate node is the node we are planning to add to
            # previously solved subpath of shorter length
            for candidate in range(n_nodes):

                # generated subset already includes the candidate node
                # here we make sure it does
                if (candidate == start) or notIn(candidate, subset):
                    continue

                # state encodes prev solved subpath of shorter length
                # we'll use it to look up the solution in the memo table
                state = subset ^ (1 << candidate)
                min_dist = float('inf')

                # tail node is last visited node in prev solved subpath
                for tail in range(n_nodes):

                    invalid_tail = (
                        tail == start
                        or tail == candidate
                        or notIn(tail, subset)
                    )
                    if invalid_tail:
                        continue

                    new_dist = (
                        M[tail][candidate]  # adding new edge
                        + memo[tail][state]  # to prev solved subpath
                    )

                    if new_dist < min_dist:
                        min_dist = new_dist

                # remember only the best path
                # from candidate through subset to the start node
                memo[candidate][subset] = min_dist


def bit_combos(r, size):
    """Generate all bit sets of given size with r bits set to 1.

    > bit_combos(1, 3)
    > [1, 2, 4]
    > # same as ['001', '010', '100']
    """

    subsets = []
    _recursive_bit_combos(0, 0, r, size, subsets)

    return subsets


def _recursive_bit_combos(set, at, r, size, L):
    if r == 0:
        L.append(set)
    else:
        for i in range(at, size):
            # flip on ith bit
            set = set | (1 << i)

            _recursive_bit_combos(set, i + 1, r - 1, size, L)

            # backtrack and flip off ith bit
            set = set & ~(1 << i)

    return


def notIn(i, subset):
    """Returns True if i-th bit in the subset is not set."""

    return (1 << i) & subset == 0


def find_min_cost(M, start_node, memo):
    """Returns minimum length of tour.

    Connects tour back to starting node and minimizes the cost.
    """

    n_nodes = len(M)

    # the end state is a bit mask with all bits set to 1
    END_STATE = (1 << n_nodes) - 1
    min_cost = float('inf')

    # the end node is the last node in solved subtour of length n_nodes
    for end_node in range(n_nodes):

        if end_node == start_node:
            continue

        # connect back to the start node and update the min cost
        cost = memo[end_node][END_STATE] + M[end_node][start_node]
        min_cost = min(min_cost, cost)

    return min_cost


def find_optimal_tour(M, start_node, memo):
    """Returns the optimal tour.

    Works backwards from end state.
    """
    n_nodes = len(M)

    last_index = start_node
    state = (1 << n_nodes) - 1  # we start at END_STATE
    tour = n_nodes*[None] + 1

    # step through all nodes in a tour, skip the start node
    for _ in range(1, n_nodes):
        optimal = None  # re-set optimal node for this step

        # find idx of optimal node at this step
        for candidate in range(0, n_nodes):

            if (candidate == start_node) or notIn(candidate, state):
                continue

            if optimal == None:
                optimal = candidate

            # TODO: i do not understand what happens here
            prev_dist = memo[optimal][state] + M[optimal][last_index]
            new_dist = memo[candidate][state] + M[candidate][last_index]

            if new_dist < prev_dist:
                optimal = candidate

        tour.append(optimal)
        state = state ^ (1 << optimal)  # exclude accounted node
        last_index = optimal

    return tour
