from mpi4py import MPI
import numpy as np
import tensorflow as tf
import pickle 

from solver.utils import sparse_dot, sparse_add, AmortizedChoice
from solver.distributed.distributed_solver import DistributedSolver


class Dvr(DistributedSolver):

    COMM_STEP_TYPE = 1
    COMP_STEP_TYPE = 2

    def __init__(self, batch_factor=1., **kwargs):
        kwargs["computation_time"] = kwargs["computation_time"] / kwargs["dataset"].N
        super(Dvr, self).__init__(**kwargs)

        self.error_computation_period = 1000
        self.iterations_multiplier = 600

        self.log.info(f"batch_factor: {batch_factor}")
        self.local_smoothnesses = self.model.get_smoothnesses(self.dataset)

        self.initialize_sigma_and_tau()

        own_smooth = np.array([self.sigma + batch_factor * np.sum(self.local_smoothnesses)])
        smooth = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(own_smooth, smooth, op=MPI.MAX) 

        min_eig, max_eig = self.get_graph_params()

        L_comm = max_eig / self.sigma
        
        self.fs_probas = np.ones((self.dataset.N,)) / self.dataset.N
        self.amortized_choice = AmortizedChoice(self.fs_probas)

        self.alpha = 2 * min_eig / smooth[0] 

        self.L_rel_array_without_alpha = np.divide(
            self.alpha * (1 + self.local_smoothnesses / self.sigma), self.fs_probas 
            )

        L_comp_buffer = np.array([max(self.L_rel_array_without_alpha)])
        max_L_comp_no_alpha = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(L_comp_buffer, max_L_comp_no_alpha, op=MPI.MAX) 

        L_comm_buffer = np.array([L_comm])
        max_L_comm = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(L_comm_buffer, max_L_comm, op=MPI.MAX) 

        L_comp_no_alpha = max_L_comp_no_alpha[0]
        L_comm = max_L_comm[0]

        L_comp =  L_comp_no_alpha

        self.p_comp = 1. / (1 + L_comm / L_comp)
        self.p_comm = 1 - self.p_comp

        # The two should be equal
        self.step_size = min(self.p_comp / L_comp, self.p_comm / L_comm) 
        
        self.rho = self.step_size * self.alpha / self.p_comp
        
        self.global_grad = np.empty(self.x.shape)
        self.nb_nodes = self.comm.size

        # actually stores z_dot_Xi
        self.z = 2 * self.dataset.y # np.zeros((self.dataset.N,)) # self.model.get_fenchel_gradient(self.x, self.dataset, step_size)
        self.last_g = [self.model.get_1d_stochastic_gradient(z, self.dataset, i) for i, z in enumerate(self.z)]
        self.x = np.zeros((self.dataset.d))
        for i, g in enumerate(self.last_g):
            sparse_add(self.x, self.dataset.X.T[i].indices, self.dataset.X.T[i].data, - g / self.sigma)

        self.X = [[x.data, x.indices] for x in self.dataset.X.T]
        
        self.log.info(f"L_comm: {L_comm}, L_comp: {L_comp}")
        self.log.info(f"p_comm: {self.p_comm}")
        self.log.info(f"Step size: {self.step_size}")

    def initialize_sigma_and_tau(self):
        self.sigma = self.model.c

    def get_graph_params(self):
        return self.graph.min_eig, self.graph.max_eig

    def update_time(self):
        pass

    def run_step(self):
        # It is important to have a coordinated seed for this.
        if self.rs.random() < self.p_comm:
            self.comm_step()
            self.current_time += self.communication_time
            self.step_type.append(self.COMM_STEP_TYPE)
        else:
            self.current_time += self.computation_time
            self.comp_step()
            self.step_type.append(self.COMP_STEP_TYPE)

    def comm_step(self):
        self.x = self.x - self.multiply_by_w(self.x) * self.step_size / (self.sigma * self.p_comm)

    def comp_step(self):
        j = self.amortized_choice.get() #j = np.random.choice(self.local_samples, p=self.fs_probas)
        rho_ij = self.rho / self.fs_probas[j]
        assert(rho_ij < 1.)
        data_j, indices_j = self.X[j]
        self.z[j] = (1 - rho_ij) * self.z[j] + rho_ij * sparse_dot(self.x, indices_j, data_j)
        new_g = self.model.get_1d_stochastic_gradient(self.z[j], self.dataset, j)
        sparse_add(self.x, indices_j, data_j, - (new_g - self.last_g[j]) / self.sigma) 
        
        self.last_g[j] = new_g


class Dvr_Cheb(Dvr):
    COMM_STEP_TYPE = 4
    COMP_STEP_TYPE = 2

    def __init__(self, **kwargs):
        super(Dvr_Cheb, self).__init__(**kwargs)
        self.log.info(f"Multi-consensus with {self.nb_comm_steps} steps")
        
    def get_graph_params(self):
        return self.graph.get_cheb_acc_constants()
    
    def comm_step(self):
        self.x = self.x - self.multiply_by_PW(self.x, self.nb_comm_steps) * self.step_size / (self.sigma * self.p_comm)
