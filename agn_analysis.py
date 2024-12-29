import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the preprocessed data
data = pd.read_csv(r"D:\AGN Research Paper\preprocessed_agn_data.csv")

# Add after loading data
data['optical_xray_ratio'] = data['r'] / data['EP_8_FLUX']
data['xray_hardness_ratio'] = (data['EP_8_FLUX'] - data['EP_1_FLUX']) / (data['EP_8_FLUX'] + data['EP_1_FLUX'])

# Separate features (exclude 'class' column)
features = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'EP_8_FLUX', 'EP_1_FLUX']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# 1. Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(X_scaled_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('agn_correlation_matrix.png')
plt.close()

# 2. Feature Distributions
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 4, i)
    sns.histplot(X_scaled_df[feature], kde=True)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.savefig('agn_feature_distributions.png')
plt.close()

# 3. PCA Analysis
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.savefig('pca_variance.png')
plt.close()

# 4. Clustering Analysis
n_clusters_range = range(2, 10)
silhouette_scores = []

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Clustering Analysis: Optimal Number of Clusters')
plt.savefig('clustering_analysis.png')
plt.close()

# 5. Optical-X-ray Relationship
plt.figure(figsize=(10, 6))
plt.scatter(data['optical_xray_ratio'], data['xray_hardness_ratio'], alpha=0.5)
plt.xlabel('Optical-to-X-ray Ratio')
plt.ylabel('X-ray Hardness Ratio')
plt.title('AGN X-ray Properties')
plt.savefig('xray_properties.png')
plt.close()

# 6. Generate Summary Statistics
summary_stats = pd.DataFrame({
    'Mean': X_scaled_df.mean(),
    'Std': X_scaled_df.std(),
    'Min': X_scaled_df.min(),
    'Max': X_scaled_df.max(),
    'Median': X_scaled_df.median()
})
summary_stats.to_csv('summary_statistics.csv')

# 7. Redshift Distribution Analysis
plt.figure(figsize=(10, 6))
sns.histplot(data['redshift'], bins=30, kde=True)
plt.xlabel('Redshift')
plt.ylabel('Count')
plt.title('AGN Redshift Distribution')
plt.savefig('redshift_distribution.png')
plt.close()

# 8. Create Results Table
results_table = pd.DataFrame({
    'Metric': ['Total AGN Sources', 'Mean Redshift', 'Median X-ray Flux', 
               'Mean Optical-X-ray Ratio', 'Mean Hardness Ratio'],
    'Value': [len(data), 
              data['redshift'].mean(),
              data['EP_8_FLUX'].median(),
              data['optical_xray_ratio'].mean(),
              data['xray_hardness_ratio'].mean()]
})
results_table.to_csv('analysis_results.csv')
