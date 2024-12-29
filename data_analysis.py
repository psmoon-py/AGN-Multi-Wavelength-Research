import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_redshift_distribution(data):
    """
    Analyze redshift distribution shown in redshift_distribution.jpg
    - Peak around z=0
    - Extended tail to z~4
    """
    stats_dict = {
        'mean_z': data['redshift'].mean(),
        'median_z': data['redshift'].median(),
        'std_z': data['redshift'].std(),
        'z_range': (data['redshift'].min(), data['redshift'].max())
    }
    
    # Calculate redshift bins for population analysis
    z_bins = [-2, 0, 1, 2, 4]
    pop_counts = pd.cut(data['redshift'], bins=z_bins).value_counts()
    
    return stats_dict, pop_counts

def analyze_pca_components(pca, X_scaled):
    """
    Analyze PCA results shown in pca_variance.jpg
    - ~80% variance explained by first 4 components
    """
    # Calculate cumulative variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for different variance thresholds
    var_thresholds = [0.7, 0.8, 0.9]
    n_components = {}
    for threshold in var_thresholds:
        n_components[threshold] = np.argmax(cum_var >= threshold) + 1
    
    # Get component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=X_scaled.columns
    )
    
    return n_components, loadings

def analyze_xray_optical_relations(data):
    """
    Analyze X-ray and optical relationships shown in xray_properties.jpg
    and correlation matrix
    """
    # Calculate key statistics for X-ray properties
    xray_stats = {
        'mean_hardness': data['xray_hardness_ratio'].mean(),
        'median_hardness': data['xray_hardness_ratio'].median(),
        'mean_opt_xray': data['optical_xray_ratio'].mean(),
        'median_opt_xray': data['optical_xray_ratio'].median()
    }
    
    # Identify outliers using IQR method
    def find_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]
        return outliers
    
    outliers = {
        'hardness_ratio': find_outliers(data['xray_hardness_ratio']),
        'opt_xray_ratio': find_outliers(data['optical_xray_ratio'])
    }
    
    return xray_stats, outliers

def analyze_optical_correlations(data):
    """
    Analyze optical band correlations shown in correlation matrix
    - Strong correlations between u,g,r,i,z
    """
    optical_bands = ['u', 'g', 'r', 'i', 'z']
    corr_matrix = data[optical_bands].corr()
    
    # Find strongest and weakest correlations
    correlations = []
    for i in range(len(optical_bands)):
        for j in range(i+1, len(optical_bands)):
            correlations.append({
                'bands': f'{optical_bands[i]}-{optical_bands[j]}',
                'correlation': corr_matrix.iloc[i,j]
            })
    
    correlations = pd.DataFrame(correlations)
    correlations = correlations.sort_values('correlation', ascending=False)
    
    return correlations

def main():
    # Load data
    data = pd.read_csv("preprocessed_agn_data.csv")
    
    # Perform analyses
    redshift_stats, redshift_pop = analyze_redshift_distribution(data)
    xray_stats, outliers = analyze_xray_optical_relations(data)
    optical_corr = analyze_optical_correlations(data)
    
    # Modify redshift statistics to split the range
    redshift_stats_df = {
        'mean_z': [redshift_stats['mean_z']],
        'median_z': [redshift_stats['median_z']],
        'std_z': [redshift_stats['std_z']],
        'z_range_min': [redshift_stats['z_range'][0]],
        'z_range_max': [redshift_stats['z_range'][1]]
    }
    
    # Save results
    pd.DataFrame(redshift_stats_df).to_csv('redshift_stats.csv', index=False)
    redshift_pop.to_csv('redshift_populations.csv')
    pd.DataFrame(xray_stats, index=[0]).to_csv('xray_stats.csv')
    optical_corr.to_csv('optical_correlations.csv')
    
    return redshift_stats_df

if __name__ == "__main__":
    results = main()
    print("Analysis complete. Results saved to CSV files.")
