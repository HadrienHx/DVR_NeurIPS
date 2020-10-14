from mpi4py import MPI
import numpy as np
import tensorflow as tf
import pickle 

from solver.distributed.distributed_solver import DistributedSolver


class GD(DistributedSolver):
    def __init__(self, batch_factor=1., **kwargs):
        super(GD, self).__init__(**kwargs)
        self.step_size = 1. / (batch_factor * np.sum(self.model.get_smoothnesses(self.dataset)) + self.model.c)

    def run_step(self):
        if self.id ==0:
            self.x -= self.step_size * self.model.get_gradient(self.x, self.error_dataset)
