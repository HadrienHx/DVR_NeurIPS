import numpy as np
import matplotlib.pyplot as plt
import logging 
import time
import tensorflow as tf
import os 


class Solver(object):
    def __init__(self, name="Main", model=None, dataset=None, seed=0, nb_epochs=0, 
    inner_precision=1e-07, max_inner_repeats=1, tb_dir=None, tb_min_error=0., 
    timestamp="", error_dataset=None, error_model=None, **kwargs):
        self.name = name
        self.model = model
        self.dataset = dataset
        if error_dataset is None:
            self.error_dataset = self.dataset
        else: 
            self.error_dataset = error_dataset
        if error_model is None:
            self.error_model = self.model
        else:
            self.error_model = error_model
        self.x = np.zeros(dataset.d)
        self.seed = seed
        self.rs = np.random.RandomState(seed=seed)
        
        self.iteration_number = 1

        self.inner_precision = inner_precision
        self.max_inner_repeats = max_inner_repeats

        self.n_inner_iters = self.dataset.N
        self.error = []
        self.log = self.get_logger()
        self.tb_min_error = tb_min_error
        self.error_computation_period = 50
        self.setup_tensorboard(tb_dir, timestamp)
        self.timestamp = int(timestamp)
        self.current_time = 0
        self.iterations_multiplier = 1
        self.time = []
        self.iteration_index = []
        self.log.info(f"seed: {self.seed}")
        self.log.info(f"Local dataset size: {self.dataset.N}")

    def get_logger(self):
        return logging.getLogger(self.name)

    def setup_tensorboard(self, tensorboard_dir, timestamp):
        self.tensorboard_dir = tensorboard_dir

        if tensorboard_dir is not None:
            self.log.info(f"Tensorboard summary writer dir: {tensorboard_dir}")
            self.summary_writer = tf.summary.create_file_writer(
                os.path.join(tensorboard_dir, timestamp, self.name))

    def solve(self, nb_epochs):
        self.update_time()
        self.log.info(f"Starting iterations, dataset shape: {self.dataset.X.shape}")
        
        nb_iterations = nb_epochs * self.iterations_multiplier
        self.log_and_report_epoch(0, nb_iterations)
        
        while self.iteration_number < nb_iterations - 1:
            self.run_step()
            self.update_time()
            self.iteration_number += 1  
            if self.iteration_number % self.error_computation_period == 0:
                self.log_and_report_epoch(self.iteration_number, nb_epochs)
             
        self.log.info(self.x[:5])

        self.log_and_report_epoch(self.iteration_number, nb_epochs)


    def log_and_report_epoch(self, i, nb_epochs):
        self.error.append(self.compute_error())
        self.time.append(self.current_time)
        self.iteration_index.append(i)
        
        self.log.info(f"Epoch {i}/{nb_epochs * self.iterations_multiplier}, error: {self.error[-1]}")
        if self.tensorboard_dir is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("error", self.error[-1], step=i+1)
                tf.summary.scalar("log_subop", np.log10(self.error[-1] - self.tb_min_error), step=int(self.current_time * self.dataset.N))
                tf.summary.scalar("time", self.current_time, step=i+1)
                tf.summary.scalar("error_time", self.error[-1], step=int(self.current_time * self.dataset.N))
            
    def compute_error(self):
        return self.error_model.compute_error(self.x, self.error_dataset)

    def update_time(self):
        self.current_time += 1
        
    def run_step(self):
        raise NotImplementedError