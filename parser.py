import argparse
import numpy as np 
import json 
import os


class Parser(object):
    def __init__(self):
        self.args = None 
        self.data_args = None 
        self.model_args = None
        self.algo_args = None
        self.config = None

    def filter_config(self, args):
        return {k: v for k,v in self.config.items() if k in args}

    def get_data_args(self):
        parser = argparse.ArgumentParser('Dataset arguments parser')

        parser.add_argument(
            '--N', type=int, default=None, help='Points per node')
        parser.add_argument(
            '--n_features', type=int, default=None, help='Dimension of the dataset (see sklearn load_svmlight)')
        parser.add_argument(
            '--length', type=int, default=-1, help='Number of bytes to read (see sklearn load_svmlight)')
        parser.add_argument(
            '--path_to_data', type=str, default='rcv1_test.binary', help='path to the dataset')
        parser.add_argument(
            '--cache_path', type=str, default=None, help='joblib cache path'
        )
        parser.add_argument(
            '--split', type=str, default='random',
            help='How to split the data, random: N random elements are chosen, fixed: elements [N * rank: N * (rank + 1)]'
        )
        # Overrides parser arguments using config file
        parser.set_defaults(**self.filter_config(
        {'N', 'path_to_data', 'cache_path', 'split', 'n_features', 'length', 'zero_based'}))
        self.data_args, _ = parser.parse_known_args()

    def get_model_args(self):
        parser = argparse.ArgumentParser(
            'Parser for model arguments. Only regularized logistic regression is supported.')
        parser.add_argument(
        '--c', type=float, default=np.power(0.1,6), help='Regularization parameter')
        parser.set_defaults(**self.filter_config({'c'}))
        self.model_args, _ = parser.parse_known_args()

    def get_algo_args(self):
        parser = argparse.ArgumentParser('Arguments for the solver')
        parser.add_argument(
            '--inner_epochs', type=int, default=10, help='DEPRECATED - Number of local solver epochs'
        )
        parser.add_argument(
            '--tb_dir', type=str, default=None, help='Where to log tensorboard'
        )
        parser.add_argument(
            '--tb_min_error', type=float, default=0., help='Min error for nice tensorboard plots'
        )
        parser.add_argument(
            '--communication_time', type=float, default=1., help='Time for one communication'
        )
        parser.add_argument(
            '--computation_time', type=float, default=1., help='Time for computing one full gradient'
        )

        # Overrides parser arguments using config file
        parser.set_defaults(**self.filter_config({
            'nb_epochs', 'tb_dir', 'tb_min_error', 'computation_time', 'communication_time'
        }))
        self.algo_args, _ = parser.parse_known_args()

    def get_main_args(self):
        parser = argparse.ArgumentParser(
            'To run this file, run mpiexec -n nb_nodes python main.py '
            'with nb_nodes the number of processors you wish to use'
        )
        parser.add_argument(
            '--seed', dest='seed', type=int, default=0, help='Random seed'
        )
        parser.add_argument(
            '--nb_epochs', type=int, default=40, help='Number of synchronized steps'
        )
        parser.add_argument(
            '--save_png', action='store_true', help='Save plot'
        )
        parser.add_argument(
            '--plot', action='store_true', help='Whether to plot the result'
        )
        parser.add_argument(
            '--output_path', type=str, default=None, help='Where to save plot data'
        )
        parser.add_argument(
            '--graph', type=str, default=None, help='Define the graph topology'
        )
        parser.add_argument(
            '--plotter_path', type=str, default=None, help='Path to other curves to show in the plot'
        )

        # Overrides parser arguments using config file
        parser.set_defaults(**self.filter_config({'nb_epochs',
        'plot', 'save_png', 'output_path', 'plotter_path', 'seed', 'graph'}))
        self.args, _ = parser.parse_known_args()

    def get_config(self):
        parser = argparse.ArgumentParser("Config file to override the other parsers")
        parser.add_argument(
            '--config_file',
            dest='config_file',
            type=str,
            default='config.json',
            help='config file',
        )
        args, _ = parser.parse_known_args()
        self.config = json.load(open(args.config_file))

    def get_args(self):
        self.get_config()
        self.get_main_args()
        self.get_data_args()
        self.get_algo_args()
        self.get_model_args()
        solvers = self.config["solvers"]
        return self.args, vars(self.data_args), vars(self.algo_args), vars(self.model_args), solvers

    def save_config(self, output_path=None, filename=None):
        path = os.path.join(output_path, filename)
        json.dump(self.config, open(path + ".conf", "w"))