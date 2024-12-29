import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# First load and prepare the data
data = pd.read_csv("preprocessed_agn_data.csv")

# Define features based on correlation matrix
# We can see strong correlations between optical bands (u,g,r,i,z)
# and between X-ray fluxes (EP_8_FLUX and EP_1_FLUX)
features = ['u', 'g', 'r', 'i', 'z', 'redshift', 'EP_8_FLUX', 'EP_1_FLUX']
X = data[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform initial clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters based on X-ray properties plot
kmeans_labels = kmeans.fit_predict(X_scaled)

from validation_analysis import validate_clustering, analyze_cluster_properties
from statistical_tests import perform_ks_test
from physical_interpretation import analyze_redshift_evolution
from detailed_clustering import optimize_clusters

# After your existing clustering code:
# Validate clustering
sil_score, ch_score = validate_clustering(X_scaled, kmeans_labels)
validation_metrics = pd.DataFrame({
    'Metric': ['Silhouette Score', 'Calinski-Harabasz Score'],
    'Value': [sil_score, ch_score]
})
validation_metrics.to_csv('validation_metrics.csv', index=False)

# Analyze cluster properties
cluster_stats = analyze_cluster_properties(data, kmeans_labels)
cluster_stats.to_csv('cluster_statistics.csv')

# Perform statistical tests
ks_results = perform_ks_test(data, kmeans_labels)
ks_results.to_csv('ks_test_results.csv')

# Analyze redshift evolution
analyze_redshift_evolution(data, kmeans_labels)

# Optimize clustering
# Replace the last two lines with:
silhouette_scores = optimize_clusters(X_scaled)
optimization_results = pd.DataFrame({
    'n_clusters': range(2, len(silhouette_scores) + 2),
    'silhouette_score': silhouette_scores
})
optimization_results.to_csv('silhouette_scores.csv', index=False)