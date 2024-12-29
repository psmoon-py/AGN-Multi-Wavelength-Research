import matplotlib.pyplot as plt
import numpy as np

def analyze_redshift_evolution(data, clusters):
    """Analyze how properties evolve with redshift"""
    plt.figure(figsize=(12, 8))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(data[mask]['redshift'], 
                   data[mask]['EP_8_FLUX'],
                   label=f'Cluster {cluster}',
                   alpha=0.6)
    plt.xlabel('Redshift')
    plt.ylabel('X-ray Flux')
    plt.yscale('log')
    plt.legend()
    plt.title('X-ray Flux vs Redshift by Cluster')
    plt.savefig('redshift_evolution.png')
    plt.close()
