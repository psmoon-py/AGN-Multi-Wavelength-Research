import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
import matplotlib.pyplot as plt

def validate_clustering(X_scaled, kmeans_labels):
    """Validate clustering results"""
    # Silhouette score
    sil_score = silhouette_score(X_scaled, kmeans_labels)
    # Calinski-Harabasz score
    ch_score = calinski_harabasz_score(X_scaled, kmeans_labels)
    
    return sil_score, ch_score

def analyze_cluster_properties(data, clusters):
    """Analyze physical properties of clusters"""
    cluster_stats = {}
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        cluster_stats[f'Cluster_{cluster}'] = {
            'mean_redshift': np.mean(data[mask]['redshift']),
            'mean_xray_flux': np.mean(data[mask]['EP_8_FLUX']),
            'mean_optical_mag': np.mean(data[mask]['r']),
            'size': np.sum(mask)
        }
    return pd.DataFrame(cluster_stats).T
