"""
* @ author Lex Sherman
* @ email alexandershershakov@gmail.com
* @ create date 2021-09-03 11:00:00
* @ modify date 2021-09-03 11:00:00
* @ desc Swarm intelligence algorithm for shortest path to visit all nodes8
"""

import string
import numpy as np
import pylab as plt
import networkx as nx
from graph import Graph
import matplotlib.pyplot as plt


def get_combinations(n_items: int):
    """ Generate all possible combinations of items
        Args:
            n_items (int): number of items
        Returns:
            combination matrix sizeof: (n_items, pow(2, n_items))
    """
    combinations = np.zeros((pow(2, n_items), n_items))
    for col in range(0, n_items):
        reps = pow(2, col)
        for row in range(reps, pow(2, n_items), 2 * reps):
            combinations[row: row+reps, col] = 1
    return combinations


def bruteforce_alg(distance_matrix: np.array):
    """ Bruteforce algorithm by trying all paths possible
    """
    node_count = distance_matrix.shape[0]
    combinations = get_combinations(node_count)
    best_distance = np.core.numeric.Inf

    s = list

    pass


class Ant:
    def __init__(self):
        pass


def swarm_intelligence_alg():
    pass


def visualize_nodes(paths_distances: np.array):
    """ Visualize nodes and paths between them
        https://stackoverflow.com/questions/18911994/visualize-distance-matrix-as-a-graph
    """
    # Create a graph
    G = nx.Graph()
    labels = {}
    for n in range(len(paths_distances)):
        for m in range(len(paths_distances)-(n+1)):
            G.add_edge(n, n+m+1)
            labels[(n, n+m+1)] = str(paths_distances[n][n+m+1])

    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=30)
    plt.show()


def main():
    # distances between all nodes
    paths_distances = np.array([
        (0,  7,  5,  3,  12, 14, 58, 69),
        (7,  0,  30, 5,  10, 1,  87, 21),
        (10, 57, 0,  32, 12, 14, 5,  6),
        (7,  12, 37, 0,  5,  31, 87, 21),
        (40, 37, 5,  3,  0,  14, 55, 67),
        (7,  15, 32, 5,  10, 0,  87, 21),
        (20, 17, 5,  43, 12, 47, 0,  63),
        (30, 27, 51, 3,  12, 14, 50, 0)
    ], dtype=np.float)

    graph = Graph()
    graph.init_from_adjacency_matrix(paths_distances)

    graph.get_node

    # plt.imshow(paths_distances)
    # plt.show()
    # bruteforce_alg(paths_distances)

    pass


if __name__ == "__main__":
    main()
