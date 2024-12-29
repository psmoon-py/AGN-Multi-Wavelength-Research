import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer  # Change from KNNImputer to SimpleImputer

# Load preprocessed data
data = pd.read_csv(r"D:\AGN Research Paper\preprocessed_agn_data.csv")

# After loading the data and before any processing
from multiwavelength_analysis import (analyze_multiwavelength_properties,
                                    analyze_wavelength_correlations)

# Feature selection based on correlation matrix
optical_features = ['u', 'g', 'r', 'i', 'z']
xray_features = ['EP_8_FLUX', 'EP_1_FLUX']
position_features = ['ra', 'dec']
features = optical_features + xray_features + ['redshift']

# Prepare features
X = data[features]

# Handle missing values using median imputation for each feature
imputer = SimpleImputer(strategy='median')

# First impute the optical features
X_optical = imputer.fit_transform(X[optical_features])
X_optical_df = pd.DataFrame(X_optical, columns=optical_features)

# Then impute the X-ray features
X_xray = imputer.fit_transform(X[xray_features])
X_xray_df = pd.DataFrame(X_xray, columns=xray_features)

# Finally impute redshift
X_redshift = imputer.fit_transform(X[['redshift']])
X_redshift_df = pd.DataFrame(X_redshift, columns=['redshift'])

# Combine all imputed features
X_imputed = pd.concat([X_optical_df, X_xray_df, X_redshift_df], axis=1)

# After feature selection and imputation
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Then perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# PCA Analysis
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.savefig('pca_variance_after_imputation.png')
plt.close()

# Clustering Analysis

# First prepare and scale the features
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Then perform clustering on the scaled data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Try different clustering algorithms
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add cluster labels to original data
data['kmeans_cluster'] = kmeans_labels
data['dbscan_cluster'] = dbscan_labels

# Generate multiwavelength analysis and visualizations
data_with_clusters = analyze_multiwavelength_properties(data)
wavelength_correlations = analyze_wavelength_correlations(data)

# Save the analysis results
data_with_clusters.to_csv('multiwavelength_results.csv', index=False)
wavelength_correlations.to_csv('wavelength_correlations.csv', index=False)
data.to_csv('agn_clustering_results.csv', index=False)

# Additional visualizations based on clustering
plt.figure(figsize=(12, 8))
for cluster in np.unique(kmeans_labels):
    mask = kmeans_labels == cluster
    plt.scatter(data[mask]['optical_xray_ratio'], 
                data[mask]['xray_hardness_ratio'],
                alpha=0.6,
                label=f'Cluster {cluster}')
plt.xlabel('Optical-to-X-ray Ratio')
plt.ylabel('X-ray Hardness Ratio')
plt.title('AGN Clusters in X-ray Property Space')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cluster_xray_properties.png')
plt.close()
