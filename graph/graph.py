import logging
import numpy as np

from graph.edge import GraphEdge


class Graph(object):
    name = None

    def __init__(self, size, seed=None, logger=None, node_weights=None,
                 weights_method=None, edges_delays=None, tau_comm=1., **kwargs):
        self.tau_comm = tau_comm
        self.size = size
        self.seed = seed
        if logger is None:
            self.log = logging.getLogger("graph")
        else:
            self.log = logger

        if seed is None:
            self.log.warning("None seed for the graph")

        self.random_state = np.random.RandomState(seed)
        self.edges = []
        self.weights = None
        self.adjacency = np.zeros((size, size))
        self.laplacian = None
        self.active = None
        self.gamma = 0
        self.Arank = None
        self.edges_delays = None
        self.incident_edges = [[] for _ in range(size)]
        self.diameter = None

        self.neighbours = [[] for _ in range(size)]
        self.edges_set = set()

    def __str__(self):
        return self.name + "_" + str(self.size)

    def build_graph(self, kwargs):
        self.build_edges(kwargs)
        self.weights = np.ones((len(self.edges),)) / len(self.edges)
        self.compute_matrices()

    def add_edge(self, i, j):
        if j < i:
            i, j = j, i

        if (i, j) not in self.edges_set:
            self.edges_set.add((i, j))
            self.edges.append(GraphEdge(len(self.edges), i, j))

            self.neighbours[i].append(j)
            self.neighbours[j].append(i)

            edge_id = len(self.edges_set) - 1
            self.incident_edges[i].append(edge_id)
            self.incident_edges[j].append(edge_id)

    def build_edges(self, kwargs):
        raise NotImplementedError

    def get_nb_edges(self):
        return len(self.edges)

    def compute_matrices(self):
        self.update_graph([1.] * len(self.edges))

    def forget_laplacian(self):
        self.laplacian = None

    def update_graph(self, mus):
        A = np.zeros((self.size, len(self.edges)), dtype=np.double)
        L = np.zeros((self.size, self.size), dtype=np.double)

        for edge in self.edges:
            mu = mus[edge.i]
            smu = np.sqrt(mu)
            A[edge.i, edge.idx] = smu
            A[edge.j, edge.idx] = - smu

            L[edge.i, edge.i] += mu
            L[edge.j, edge.j] += mu
            L[edge.i, edge.j] -= mu
            L[edge.j, edge.i] -= mu

        self.A = A
        self.laplacian = L
        w, _ = np.linalg.eigh(self.laplacian)
        w = np.sort([np.real(x) for x in w])

        self.min_eig = w[1] 
        self.max_eig = w[-1]

        assert(self.min_eig > 0)

        self.gamma = self.min_eig / self.max_eig

    def get_cheb_acc_constants(self):
        k = int(np.ceil(1. / np.sqrt(self.gamma)))

        sq_gamma = np.sqrt(self.gamma)

        c1 = (1 - sq_gamma) / (1 + sq_gamma)
        c2 = (1 + self.gamma) / (1 - self.gamma)
        c3 = 2 / ((1 + self.gamma) * self.max_eig)

        a0 = 1. 
        a1 = c2 
        X0 = np.eye(len(self.laplacian))
        X1 = c2 * (X0 - c3 * self.laplacian)

        for _ in range(1, k):
            a1, a0 = [2 * c2 * a1 - a0, a1]
            temp_X1 = 2 * c2 * (X1 - c3 * self.laplacian @ X1) - X0
            X0 = X1
            X1 = temp_X1

        PW = np.eye(len(self.laplacian)) - X1 / a1

        w, _ = np.linalg.eigh(PW)
        w = np.sort([np.real(x) for x in w])

        min_eig = w[1] 
        max_eig = w[-1]

        assert(min_eig > 0)

        return min_eig, max_eig

