import numpy as np

from solver.distributed.distributed_solver import DistributedSolver


class Extra_Cata(DistributedSolver):
    STEP_TYPE = 0

    def __init__(self, batch_factor=1., tau=0., inner_iters=100, **kwargs):
        super(Extra_Cata, self).__init__(**kwargs)
        self.number_inner_iterations = inner_iters
        self.log.info(f"batch_factor: {batch_factor}")
        self.L = self.model.c + batch_factor * np.sum(self.model.get_smoothnesses(self.dataset))
        gamma = self.graph.gamma
        self.tau = self.L * gamma
        assert(self.tau > 0)
        q = self.model.c / (self.model.c + self.tau)
        self.theta = np.sqrt(q)

        self.beta =  self.tau + self.L 
        self.alpha = 1. / self.beta
        
        self.log.info(f"tau: {self.tau}")

        self.nb_nodes = self.comm.size

        self.y = np.copy(self.x)
        self.v = np.zeros(self.x.shape)

        self.last_outer_x = np.copy(self.x) 
        self.coeff = self.theta * (1 - self.theta) / (np.power(self.theta, 2) + self.theta)
        self.last_comm_step = 0.5 * self.beta * self.communication_step(self.x)

        self.number_inner_iterations = int(np.ceil(0.2 * np.log(self.L / (self.model.c * gamma)) / gamma))
        self.log.info(f"inner iters: {self.number_inner_iterations}")
        self.log.info(f"kappa: {self.L / self.model.c};  coeff: {self.coeff}")

    def communication_step(self, x):
        return self.multiply_by_w(x) / self.graph.max_eig
        # return self.multiply_by_PW(x, self.nb_comm_steps) / self.max_eig_cheb

    def update_time(self):
        self.current_time += self.computation_time + self.nb_comm_steps * self.communication_time
        self.step_type.append(self.STEP_TYPE)

    def outer_step(self):
        self.y = (1 + self.coeff) * self.x - self.coeff * self.last_outer_x
        self.last_outer_x = np.copy(self.x)

    def run_step(self):
        if self.iteration_number % self.number_inner_iterations == 0:
            self.outer_step()
        
        local_grad = self.model.get_gradient(self.x, self.dataset) + self.tau * (self.x - self.y)

        self.x = self.x - self.alpha * (local_grad + self.v + self.last_comm_step)
        self.last_comm_step = np.copy(0.5 * self.beta * self.communication_step(self.x)) 
        self.v = self.v + self.last_comm_step
