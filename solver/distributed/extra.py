from mpi4py import MPI
import numpy as np
import tensorflow as tf
import pickle 

from solver.distributed.distributed_solver import DistributedSolver


class Extra(DistributedSolver):
    def __init__(self, batch_factor=1., **kwargs):
        super(Extra, self).__init__(**kwargs)
        self.log.info(f"batch_factor: {batch_factor}")
        self.step_size = 1. / (self.model.c + batch_factor * np.sum(self.model.get_smoothnesses(self.dataset)))

        self.nb_nodes = self.comm.size
        self.last_x = np.copy(self.x)
        self.last_g = self.model.get_gradient(self.x, self.dataset)

        self.x = self.x - self.communication_step(self.x) - self.step_size * self.last_g

    def communication_step(self, x):
        return self.multiply_by_w(x) / self.graph.max_eig

    def run_step(self):
        delta_x = 2 * self.x - self.last_x
        local_grad = self.model.get_gradient(self.x, self.dataset)

        new_x = delta_x - 0.5 * self.communication_step(delta_x) - (
                self.step_size * (local_grad - self.last_g)
        )
        self.last_x = self.x
        self.x = new_x 
        self.last_g = local_grad


class Extra_Cheb(Extra):
    def __init__(self, **kwargs):
        super(Extra_Cheb, self).__init__(**kwargs)
        self.log.info(f"Multi-consensus with {self.nb_comm_steps} steps")

    def communication_step(self, x):
        return self.multiply_by_PW(x, self.nb_comm_steps) 

    def update_time(self):
        self.current_time += self.computation_time + self.nb_comm_steps * self.communication_time
        self.step_type.append(0)