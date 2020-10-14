from joblib import Memory
from sklearn.datasets import load_svmlight_file
import numpy as np 

from dataset.dataset import Dataset 
from dataset.loader import Loader 


load_svmlight_kwargs = {
    "n_features", "dtype", "multilabel", "zero_based",
    "query_id", "offset", "length"
}


class LIBSVM_Loader(Loader):
    def __init__(self,  **kwargs):
        if "length" in kwargs:
            self.bytes_length = kwargs["length"]
        super(LIBSVM_Loader, self).__init__(**kwargs)

    def load(self, path_to_data=None, comm_size=None, **kwargs):
        restricted_kwargs = {
            k: v for k,v in kwargs.items() 
            if k in load_svmlight_kwargs
        }

        @self.mem_loader.cache
        def get_data(path_to_data, **kwargs):
            return load_svmlight_file(path_to_data, **kwargs)[:2]

        X, y = get_data(path_to_data, **restricted_kwargs)
        d = Dataset(X=X.T, y=y.T, seed=self.seed)

        if self.nb_samples is not None and comm_size is not None:
            tot_N = comm_size * self.nb_samples
            if d.N > tot_N:
                d.truncate(tot_N, 0)
            if d.N < tot_N:
                self.nb_samples = int(np.floor(d.N / comm_size))
                d.truncate(self.nb_samples * comm_size, 0)

        return d