



from scipy.stats import wasserstein_distance, energy_distance
import numpy as np
import ot
import matplotlib.pyplot as plt
from ot import plot
from ot.datasets import make_1D_gauss as gauss
from scipy.stats import distributions
from ML_tools.sklearn_tools import kolmogorov_smirnov_test_statistic



def compute_emd(bin_centers, distrib1, distrib2):
    
    sample1 = np.concatenate([bin_centers.reshape(-1,1), distrib1.values.reshape(-1,1)], axis=1)
    sample2 = np.concatenate([bin_centers.reshape(-1,1), distrib2.values.reshape(-1,1)], axis=1)
        
    x = bin_centers
    M = ot.dist(sample1, sample2)
    M /= M.max()

    em_distance = ot.emd2(a=np.ones(len(sample1)),b=np.ones(len(sample1)),M=M)
    
    return em_distance



def kolmogorov_smirnoff_test_weighted(data1, data2, wei1, wei2, alternative='two-sided'):

    d = kolmogorov_smirnov_test_statistic(data1, data2, wei1, wei2)

    """ Code from scipy.stats.ks_2samp"""
    
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the larger of (n1, n2)
        expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
        prob = np.exp(expt)
        
    return d, prob




     