from numba import njit
import numpy as np
from mpi4py import MPI


@njit
def sparse_dot(x, y_indptr, y_data):
    s = 0. 
    for i in range(len(y_indptr)):
        s += x[y_indptr[i]] * y_data[i]
    return s 

@njit
def sparse_add(x, y_indptr, y_data, coeff):
    for i in range(len(y_indptr)):
        x[y_indptr[i]] += coeff * y_data[i]

def agree_on(comm, x, op):
    new_var = np.empty((1,), dtype=np.float64)
    var = np.array([x])
    comm.Allreduce(var, new_var, op=op)
    return var[0]


class AmortizedChoice(object):
    def __init__(self, p, size=1000, seed=None):
        self.p = p
        self.size = size
        self.local_samples = np.arange(len(p))
        self.random_indices = None
        self.idx = 0
        self.rs = np.random.RandomState(seed)

        self.generate_new_indices()

    def generate_new_indices(self):
        self.random_indices = self.rs.choice(self.local_samples,
        p=self.p, size=self.size)

    def get(self):
        if self.idx == len(self.random_indices):
            self.generate_new_indices()
            self.idx = 0

        new_idx = self.random_indices[self.idx]
        self.idx += 1
        return new_idx