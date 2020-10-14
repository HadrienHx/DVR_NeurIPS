## Code to run the DVR algorithm

This repository contains the code for the NeurIPS paper "Dual-Free Stochastic Decentralized Optimization with Variance Reduction", which introduces the DVR algorithm. Details and theory for the algorithm can be found in the paper (DVR.pdf file).  

# Requirements

Create a new python 3 environment:

`conda create --name dvr_env python=3.8`

Install the following packages:

`conda install -c conda-forge openmpi`

`conda install matplotlib numba scikit-learn mpi4py numpy scipy tensorflow`


# Run the code

To run the code, and plot the results, use the command:

`mpirun -n nb_nodes python main.py --plot`

with nb_nodes the number of processors on the 2D grid (4, 9, 16...)

For instance:

`mpirun -n 4 python main.py --plot`

# Configuration

By default, DVR is run using the `config.json` file in the current directory. Another file can be specified using the `--config_file` option. A sample configuration file is provided that the user can modify to test different options.

In particular, it is necessary to change the `path_to_data` and `output_path` options to specify the dataset that should be used. 
    
