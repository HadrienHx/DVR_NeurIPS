from mpi4py import MPI
import numpy as np
import tensorflow as tf
import pickle 

from solver.distributed.distributed_solver import DistributedSolver


class Nids(DistributedSolver):
    def __init__(self, batch_factor=1., **kwargs):
        super(Nids, self).__init__(**kwargs)
        self.step_size = 1. / (self.model.c + batch_factor * np.sum(self.model.get_smoothnesses(self.dataset)))

        self.global_grad = np.empty(self.x.shape)
        self.nb_nodes = self.comm.size
        self.last_x = np.copy(self.x)
        self.last_g = self.model.get_gradient(self.x, self.dataset)

        self.x = self.x - self.multiply_by_w(self.x) / self.graph.max_eig - self.step_size * self.last_g

    def comm_step(self, x):
        return x - 0.5 * self.multiply_by_w(x) / self.graph.max_eig

    def run_step(self):
        local_grad = self.model.get_gradient(self.x, self.dataset)
        new_x = self.comm_step(
            2 * self.x - self.last_x - self.step_size * (local_grad - self.last_g)
        )
        self.last_x = self.x
        self.x = new_x 
        self.last_g = local_grad
