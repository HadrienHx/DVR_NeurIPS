from numba import njit
from numba.typed import List
import numpy as np
import scipy.special
# from scipy.special import expit
from numba import njit

vec_logit = np.vectorize(lambda x: - np.logaddexp(0, -x))

@njit
def expit(x):
    if x > 0:
        return 1. / (1 + np.exp(-x))
    return np.exp(x) / (1 + np.exp(x))

class LogisticRegression(object):
    name = "LR"

    def __init__(self, c=0.):
        self.c = c
        
    def get_L(self, dataset):
        @njit 
        def get_L(indptr, data):
            L = 0
            for i in range(len(indptr) - 1):
                L = max(L, np.sum(np.power(data[indptr[i]:indptr[i+1]], 2)))
            return L
        return 0.25 * get_L(dataset.X.T.indptr, dataset.X.T.data) / dataset.N

    def get_smoothnesses(self, dataset):
        @njit 
        def get_L(indptr, data):
            return np.array([
                0.25 * np.sum(np.power(data[indptr[i]:indptr[i+1]], 2))
                for i in range(len(indptr) - 1)
            ])
        return get_L(dataset.X.T.indptr, dataset.X.T.data) / dataset.N

    def get_gradient(self, theta, dataset):
        s = dataset.X @ (dataset.y * scipy.special.expit(- dataset.y * (dataset.X.T @ theta)))
        return self.c * theta - s / dataset.N

    def get_1d_stochastic_gradient(self, theta_dot_x, dataset, i):
        return  - dataset.y[i] * scipy.special.expit(- dataset.y[i] * theta_dot_x) / dataset.N

    def compute_error(self, theta, dataset):
        s = np.sum(
            # np.log(scipy.special.expit(dataset.y * (dataset.X.T @ theta)))
            # Using vec_logit is slower but more precise
            vec_logit(dataset.y * (dataset.X.T @ theta))
        ) / dataset.N
        return 0.5 * self.c * (theta @ theta) - s

    def get_global(self, comm_size):
        return LogisticRegression(self.c * comm_size)
