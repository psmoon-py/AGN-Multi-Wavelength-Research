import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def analyze_multiwavelength_properties(data):
    """Analyze relationships between optical and X-ray properties"""
    
    # Create a copy of the data to avoid modifying the original
    data_copy = data.copy()
    
    # Handle zeros and negative values before log calculation
    epsilon = 1e-10  # Small constant to avoid log(0)
    data_copy['r'] = data_copy['r'].clip(lower=epsilon)
    data_copy['EP_8_FLUX'] = data_copy['EP_8_FLUX'].clip(lower=epsilon)
    
    # Create optical color indices
    data_copy['u_g'] = data_copy['u'] - data_copy['g']
    data_copy['g_r'] = data_copy['g'] - data_copy['r']
    data_copy['r_i'] = data_copy['r'] - data_copy['i']
    data_copy['i_z'] = data_copy['i'] - data_copy['z']
    
    # Calculate ratios with handling for invalid values
    data_copy['xray_optical_ratio'] = np.log10(data_copy['EP_8_FLUX']/data_copy['r'])
    
    # Calculate X-ray hardness ratio
    data_copy['xray_hardness_ratio'] = (data_copy['EP_8_FLUX'] - data_copy['EP_1_FLUX']) / (data_copy['EP_8_FLUX'] + data_copy['EP_1_FLUX'])
    
    # Select features for clustering
    features = ['u_g', 'g_r', 'xray_optical_ratio', 'xray_hardness_ratio']
    X = data_copy[features]
    
    # Handle missing values using median imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data_copy['cluster'] = kmeans.fit_predict(X_scaled)
    
    return data_copy
    
    # Visualizations
    
    # 1. Optical-Xray Color-Color Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data['u_g'], data['xray_optical_ratio'], 
                         c=data['cluster'], cmap='viridis')
    plt.xlabel('u-g Color')
    plt.ylabel('log(X-ray/Optical) Ratio')
    plt.title('AGN Optical-X-ray Color Distribution')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('optical_xray_colors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. X-ray Hardness vs Optical Color
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data['g_r'], data['xray_hardness_ratio'],
                         c=data['redshift'], cmap='plasma')
    plt.xlabel('g-r Color')
    plt.ylabel('X-ray Hardness Ratio')
    plt.title('X-ray Hardness vs Optical Color')
    plt.colorbar(scatter, label='Redshift')
    plt.savefig('xray_hardness_optical_color.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Multi-wavelength Parameter Space
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data['u_g'], data['xray_optical_ratio'], 
                        data['xray_hardness_ratio'], c=data['redshift'],
                        cmap='plasma')
    ax.set_xlabel('u-g Color')
    ax.set_ylabel('log(X-ray/Optical) Ratio')
    ax.set_zlabel('X-ray Hardness Ratio')
    plt.colorbar(scatter, label='Redshift')
    plt.title('AGN Multi-wavelength Parameter Space')
    plt.savefig('multiwavelength_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return data

def analyze_wavelength_correlations(data):
    """Analyze correlations between different wavelength bands"""
    
    # Calculate correlation statistics
    optical_bands = ['u', 'g', 'r', 'i', 'z']
    xray_bands = ['EP_8_FLUX', 'EP_1_FLUX']
    
    # Create empty list to store correlations
    correlation_list = []
    
    # Calculate correlations
    for opt in optical_bands:
        for xray in xray_bands:
            corr = np.corrcoef(data[opt], data[xray])[0,1]
            correlation_list.append({
                'optical_band': opt,
                'xray_band': xray,
                'correlation': corr
            })
    
    # Convert list to DataFrame
    correlations = pd.DataFrame(correlation_list)
    
    # Visualize wavelength correlations
    plt.figure(figsize=(10, 6))
    for xray in xray_bands:
        corr_vals = correlations[correlations['xray_band'] == xray]['correlation']
        plt.plot(optical_bands, corr_vals, 'o-', label=xray)
    plt.xlabel('Optical Band')
    plt.ylabel('Correlation with X-ray')
    plt.title('Optical-X-ray Band Correlations')
    plt.legend()
    plt.grid(True)
    plt.savefig('wavelength_correlations.png')
    plt.close()
    
    return correlations
