import numpy as np
import pandas as pd

from iterative_dp.pep import PEP, MyMarginals
from iterative_dp.Util import data_pub_sampling
from iterative_dp.mbi import Dataset, Domain


class PEPSynth:
    def __init__(self, epsilon, delta, marginal=3, workload=32, workload_seed=0, T=10, iters=1000):
        self.epsilon = epsilon
        self.delta = delta
        self.marginal = marginal
        self.workload = workload
        self.workload_seed = workload_seed
        self.T = T
        self.iters = iters
        self.is_fitted = False

    def _randomKway(self, data):
        config = data.nunique().to_dict()
        domain = Domain(config.keys(), config.values())
        domain_max = max(domain.config.values())
        dtype = data_pub_sampling.get_min_dtype(domain_max)
        data = data.astype(dtype)
        dataset = Dataset(data, domain)
        return dataset, data_pub_sampling.randomKwayData(dataset, self.workload,
                                                         self.marginal, self.workload_seed)


    def fit(self, data):
        self.dataset, self.workloads = self._randomKway(data)
        self.my_marginals = MyMarginals(self.dataset.domain, self.workloads)
        self.ew_algorithm = PEP(data_domain=self.dataset.domain,
                                my_marginals=self.my_marginals,
                                max_iters=self.iters)
        self.A_last = self.ew_algorithm.generate(self.dataset, self.T, self.epsilon, self.delta)
        self.is_fitted= True

    def sample(self, n_records):
        assert self.is_fitted, 'Synth was not fitted it.'
        index_array = np.arange(len(self.A_last))
        indices = np.random.choice(index_array, n_records,
                                   replace=True, p=self.A_last)
        sample_col_values = np.unravel_index(indices, self.dataset.domain.shape)
        samples = list(zip(*sample_col_values))
        return pd.DataFrame(samples, columns=list(self.dataset.domain.config.keys()))

    
    def fit_sample(self, data, n_records=None):
        n_records = len(data) if n_records is None else n_records
        self.fit(data)
        samples = self.sample(n_records)
        return samples
