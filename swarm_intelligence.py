"""
* @ author Lex Sherman
* @ email alexandershershakov@gmail.com
* @ create date 2021-09-03 11:00:00
* @ modify date 2021-09-03 11:00:00
* @ desc Swarm intelligence algorithm for shortest path to visit all nodes8
"""

import numpy as np
import pylab as plt
import networkx as nx
from graph import Graph
import matplotlib.pyplot as plt
from collections import deque
import copy


# def bfs_alg(graph: Graph):
#     """ Bruteforce algorithm by trying all paths possible
#     """
#     class Node:
#         def __init__(self, vertices, length=0):
#             self.vertices = vertices  # visited vertices
#             self.length = length      # path length between vertices

#     best_distance = np.core.numeric.Inf
#     Q = deque
#     Q.append(Node(graph.vertices[0], 0))
#     while Q:
#         vertex = Q.popleft()


#     for node in range(0, graph.vertices_no):

#     pass


def bfs_alg(node_distances: np.array):
    """ Bruteforce algorithm by trying all paths possible
    """
    assert(node_distances.shape[0] == node_distances.shape[1])
    n_vertices = node_distances.shape[0]

    class Path:
        def __init__(self, vertices: list, length=0):
            self.vertices = vertices  # visited vertices
            self.length = length      # path length between vertices

        def get_last_node(self):
            return self.vertices[len(self.vertices) - 1]

    def get_children(path: Path) -> list:
        """ Returns all children with path != 0
        """
        assert(node_distances.shape[0] == node_distances.shape[1])
        children = []
        for i in range(node_distances.shape[0]):
            dist = node_distances[path.get_last_node()][i]
            if (i not in path.vertices) and (dist != 0):
                n = Path(path.vertices.copy(), path.length)
                n.vertices.append(i)
                n.length += dist
                children.append(n)
        return children

    Q = deque([Path([0], 0)])
    results = list()
    while Q:
        path = Q.popleft()
        if len(path.vertices) < n_vertices:
            children = get_children(path)
            for child in children:
                Q.append(child)
        else:
            results.append(path)

    newlist = sorted(results, key=lambda k: k.length)
    print(f"len: {newlist[0].length}, {newlist[0].vertices}")


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

    # plt.imshow(paths_distances)
    # plt.show()
    bfs_alg(paths_distances)

    pass


if __name__ == "__main__":
    main()
