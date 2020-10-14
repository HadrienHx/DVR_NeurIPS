from mpi4py import MPI
import numpy as np
import tensorflow as tf
import pickle 

from solver.utils import sparse_dot, sparse_add, AmortizedChoice
from solver.distributed.distributed_solver import DistributedSolver


class GT_SAGA(DistributedSolver):
    def __init__(self, batch_factor=1., **kwargs):
        kwargs["computation_time"] = kwargs["computation_time"] / kwargs["dataset"].N
        super(GT_SAGA, self).__init__(**kwargs)

        self.error_computation_period = 1000
        self.iterations_multiplier = 100

        self.sigma = self.model.c 
        self.log.info(f"batch_factor: {batch_factor}")
        self.local_smoothnesses = batch_factor * self.model.get_smoothnesses(self.dataset)
        own_smooth = np.array([self.sigma + 0.01 * np.sum(self.local_smoothnesses)])
        smooth = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(own_smooth, smooth, op=MPI.MAX) 
        
        self.fs_probas = np.ones((self.dataset.N,)) / self.dataset.N
        self.amortized_choice = AmortizedChoice(self.fs_probas)

        self.alpha = 100.

        # actually stores z_dot_Xi
        self.g_table = [self.model.get_1d_stochastic_gradient(0, self.dataset, i) for i in range(self.dataset.N)]
        self.x = np.zeros((self.dataset.d))
        self.g = np.zeros((self.dataset.d))
        self.y = np.zeros((self.dataset.d))
        self.g_bar = np.zeros((self.dataset.d))
        
        for i, g in enumerate(self.g_table):
            sparse_add(self.g_bar, self.dataset.X.T[i].indices, self.dataset.X.T[i].data, g / self.dataset.N)

        self.X = [[x.data, x.indices] for x in self.dataset.X.T]

    def update_time(self):
        pass

    def run_step(self):
        self.current_time += 2 * self.communication_time
        self.current_time += self.computation_time
        self.comp_step()
        self.step_type.append(3)

    def comm_step(self, x):
        return x - self.multiply_by_w(x) / self.graph.max_eig   

    def comp_step(self):
        self.x = self.comm_step(self.x) - self.alpha * self.y
        j = self.amortized_choice.get()
        data_j, indices_j = self.X[j]
        new_g_j = self.model.get_1d_stochastic_gradient(self.dataset.X.T[j].dot(self.x), self.dataset, j)[0]
        sparse_add(self.g_bar, indices_j, data_j, new_g_j - self.g_table[j])
        new_g = np.copy(self.g_bar)
        sparse_add(self.g_bar, indices_j, data_j, (new_g_j - self.g_table[j]) * (- 1 + 1. / self.dataset.N))
        self.y = self.comm_step(self.y) + new_g - self.g

        self.g_table[j] = new_g_j 
        self.g = new_g

