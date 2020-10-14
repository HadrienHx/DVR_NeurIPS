import numpy as np
from numba import njit
from numba.typed import List


class Dataset(object):
    def __init__(self, X=None, y=None, seed=None, **kwargs):
        self.X = X
        self.y = y
        self.d, self.N = self.X.shape # X is a sparse matrix of shape (nb_features, nb_samples)
        self.seed = seed
        self.rs = np.random.RandomState(seed=seed)

    def truncate(self, N, start):
        if N < self.N:
            self.X = self.X.T[start:start + N].T
            self.y = self.y[start:start + N]
            self.N = self.X.shape[1]

    def subsample(self, N):
        if N < self.N: 
            new_args = list(range(self.N))
            self.rs.shuffle(new_args)
            new_args = new_args[:N]

            self.X = self.X.T[new_args].T
            self.y = self.y[new_args]
            self.N = self.X.shape[1]

    def get_subsampled(self, N=None):
        if N is None:
            return self
        new_d = Dataset(X=self.X, y=self.y, seed=self.seed)
        new_d.subsample(N)
        return new_d

    def get_truncated(self, rank, comm_size):
        data_copy = Dataset(X=self.X, y=self.y, seed=self.seed)
        loc_N = int(data_copy.N /comm_size)
        data_copy.truncate(loc_N, rank * loc_N)
        return data_copy