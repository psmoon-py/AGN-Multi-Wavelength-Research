from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def perform_ks_test(data, clusters):
    """Perform KS-test between clusters"""
    n_clusters = len(np.unique(clusters))
    properties = ['redshift', 'EP_8_FLUX', 'r']
    ks_results = {}
    
    for prop in properties:
        ks_results[prop] = {}
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                stat, pval = stats.ks_2samp(
                    data[clusters == i][prop],
                    data[clusters == j][prop]
                )
                ks_results[prop][f'C{i}_vs_C{j}'] = {'statistic': stat, 'p-value': pval}
    
    return pd.DataFrame(ks_results)
