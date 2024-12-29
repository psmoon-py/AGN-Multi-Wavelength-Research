from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def optimize_clusters(X_scaled, max_clusters=10):
    """Find optimal number of clusters"""
    silhouette_scores = []
    for n in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters+1), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal Number of Clusters')
    plt.savefig('cluster_optimization.png')
    plt.close()
    
    return silhouette_scores
