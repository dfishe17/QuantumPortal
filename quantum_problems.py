import numpy as np

def get_maxcut_qubo(adjacency_matrix):
    """
    Generate QUBO for MAX-CUT problem.
    """
    n = len(adjacency_matrix)
    Q = {}
    
    # Generate QUBO coefficients
    for i in range(n):
        for j in range(n):
            if i == j:
                Q[(i, i)] = 0
                for k in range(n):
                    if k != i:
                        Q[(i, i)] += adjacency_matrix[i][k] / 4.0
            elif j > i:
                if adjacency_matrix[i][j] != 0:
                    Q[(i, j)] = -adjacency_matrix[i][j] / 4.0
    
    return Q

def get_graph_coloring_qubo(adjacency_matrix, num_colors):
    """
    Generate QUBO for Graph Coloring problem.
    """
    n = len(adjacency_matrix)
    Q = {}
    
    # Penalty weights
    A = 1.0  # one color per node constraint
    B = 2.0  # adjacent nodes different colors constraint
    
    # One color per node constraint
    for i in range(n):
        for c1 in range(num_colors):
            v1 = i * num_colors + c1
            # Diagonal terms
            Q[(v1, v1)] = -A
            # Cross terms
            for c2 in range(c1 + 1, num_colors):
                v2 = i * num_colors + c2
                Q[(v1, v2)] = 2 * A
    
    # Adjacent nodes different colors constraint
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] == 1:
                for c in range(num_colors):
                    v1 = i * num_colors + c
                    v2 = j * num_colors + c
                    Q[(v1, v2)] = B
    
    return Q
