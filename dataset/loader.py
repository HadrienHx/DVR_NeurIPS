from joblib import Memory
import numpy as np


class Loader(object):
    def __init__(self, cache_path=None, N=None,
    rank=0, seed=None, **kwargs):
        self.mem_loader = Memory(cache_path)
        self.nb_samples = N
        self.seed = seed

    def load(self, **kwargs):
        raise NotImplementedError
