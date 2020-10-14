import logging
import numpy as np
from graph.graph import Graph


class Grid(Graph):
    name = "grid"

    def __init__(self, length, **kwargs):
        sqrt_l = int(np.floor(np.sqrt(length)))
        l2 = int(np.power(sqrt_l, 2))
        assert(l2 == length)
        super().__init__(l2, **kwargs)
        self.build_graph(sqrt_l)
        self.diameter = 2 * (sqrt_l - 1)

    def build_edges(self, length):
        neighbours_x = ([-1, 1], lambda current, delta: -1 < (current % length) + delta < length)
        neighbours_y = ([-length, length], lambda current, delta: -1 < current + delta < self.size)
        for i in range(self.size):
            for neighbors, condition in (neighbours_x, neighbours_y):
                for neigh in neighbors:
                    if condition(i, neigh):
                        self.add_edge(i, i + neigh)


class CompleteGraph(Graph):
    name = "complete"

    def __init__(self, length, **kwargs):
        super().__init__(length, **kwargs)
        self.build_graph(length)
        self.diameter = 1

    def build_edges(self, length):
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self.add_edge(i, j)


class Ring(Graph):
    name = "ring"

    def __init__(self, length, **kwargs):
        super().__init__(length, **kwargs)
        self.build_graph(length)
        self.diameter = int(length / 2)
        
    def build_edges(self, length):
        for i in range(self.size - 1):
            self.add_edge(i, i+1)
        self.add_edge(self.size - 1, 0)


class ErdosRenyi(Graph):
    name = "erdos_renyi"
    def __init__(self, length, p=0.1, **kwargs):
        super(ErdosRenyi, self).__init__(length, **kwargs)
        self.p = p
        self.build_graph(length)

    def build_edges(self, length):
        # Too big but not a problem.
        random_edges = self.random_state.random(size=(self.size, self.size))
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if random_edges[i, j] < self.p: 
                    self.add_edge(i, j)


graph_classes = {
    "grid": Grid,
    "ring": Ring,
    "complete": CompleteGraph,
    "erdos": ErdosRenyi
}

def get_graph_class(name):
    return graph_classes[name]
