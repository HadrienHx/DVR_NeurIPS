from numba import njit
import scipy.sparse as sparse
import numpy as np
import logging

import mpi4py
import argparse
import time 
import os 


from dataset.libsvm_loader import LIBSVM_Loader 
from models.logistic_regression import LogisticRegression
from solver.distributed.solvers import AvailableSolvers
from graph.basic_graphs import get_graph_class
from plotter import Plotter
from parser import Parser 


# Retrieve args
args_parser = Parser()
args, data_args, algo_args, model_args, solvers = args_parser.get_args()

# Identify node
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()


# Set up loggers / plotters
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"Main {rank}")

plotter = Plotter(filename=args.plotter_path)

# Load model and dataset
model = LogisticRegression(**model_args)

error_model = model.get_global(comm.size)
global_dataset = LIBSVM_Loader(**data_args, seed=args.seed, rank=rank).load(**data_args, comm_size=comm.size)
local_dataset = global_dataset.get_truncated(rank, comm.size)
if rank > 0:
    global_dataset = None

# Build graph
graph = get_graph_class(args.graph)(comm.size, seed=args.seed, logger=log)
log.info(graph)

# Name the run
filename = str(time.time()).split(".")[0]
timestamp = np.array([int(filename)])
comm.Bcast(timestamp, root=0)
filename = str(timestamp[0])

for solver_name, solver_args in solvers:
    # Retrieve solver (server / worker)
    solver_class = AvailableSolvers.get(solver_name)
    solver = solver_class(name=solver_name, comm=comm,
                model=model, error_model=model, graph=graph,
                dataset=local_dataset, error_dataset=global_dataset,
                seed=args.seed, timestamp=filename, **algo_args, **solver_args)

    # Run algorithm
    solver.solve(args.nb_epochs)

    # Servers stores results for plotting
    if rank == 0:
        plotter.add_curve(solver_name, solver.error, solver.time, solver_args,
        [solver.step_type, solver.iteration_index])

# Server plots and saves results to disk
if rank == 0:
    plotter.plot(show=args.plot)
    plotter.save(filename=filename, output_path=args.output_path, save_png=args.save_png)
    args_parser.save_config(filename=filename, output_path=args.output_path)
    log.info("Finished execution")