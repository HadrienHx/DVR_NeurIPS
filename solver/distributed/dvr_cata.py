from mpi4py import MPI
import numpy as np
import tensorflow as tf
import pickle 

from solver.utils import sparse_dot, sparse_add, AmortizedChoice
from solver.distributed.distributed_solver import DistributedSolver
from solver.distributed.dvr import Dvr_Cheb, Dvr


class Dvr_Cata(Dvr_Cheb):
    def __init__(self, tau=0., inner_iters=100, **kwargs):
        super(Dvr_Cata, self).__init__(**kwargs)
        self.number_inner_iterations = int(1.0 * np.ceil(self.dataset.N / self.p_comp))
        q = self.model.c / (self.model.c + self.tau)
        self.theta = np.sqrt(q)

        self.log.info(f"tau: {self.tau}, n_iters: {self.number_inner_iterations}")

        self.y = np.copy(self.x)
        # Because y_0 = x_0
        self.x += self.tau * self.y / self.sigma
        self.last_outer_x = np.copy(self.x)

    def initialize_sigma_and_tau(self):
        self.tau = (self.model.c + np.sum(self.local_smoothnesses)) / len(self.local_smoothnesses)
        self.sigma = self.model.c + self.tau

    def outer_step(self):
        # Useless because constant choice of theta (\sqrt(q) in the paper)
        new_theta = self.theta
        coeff = self.theta * (1 - self.theta) / (np.power(self.theta,2) + new_theta)
        self.theta = new_theta

        new_y = (1 + coeff) * self.x - coeff * self.last_outer_x
        self.last_outer_x = self.x

        # sigma contains sigma + tau already !
        self.x = self.x + self.tau * (new_y - self.y) / self.sigma
        self.y = new_y

    def run_step(self):
        # self.iteration_number starts at 1
        if self.iteration_number % self.number_inner_iterations == 0:
            self.outer_step()
        super(Dvr_Cata, self).run_step()
