import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# SDSS data loading part
sdss_data = pd.read_csv("D:\\sdss_dr17_agn_sample.csv", 
                       sep=',',  # Change to comma separator
                       skiprows=[0,1],  # Skip the header rows
                       names=['objid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'class', 'redshift', 'zwarning'])

# Convert string columns to numeric
numeric_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'zwarning']
for col in numeric_columns:
    sdss_data[col] = pd.to_numeric(sdss_data[col], errors='coerce')

# Print SDSS data info
print("SDSS data shape:", sdss_data.shape)
print("\nSDSS first few rows:")
print(sdss_data.head())

# Load XMM-Newton data and print its structure
with fits.open("D:\\4XMM_DR9cat_v1.0.fits\\4xmmdr9_191129.fits") as hdul:
    print("\nXMM-Newton data structure:")
    print(hdul.info())
    print("\nColumn names:", hdul[1].columns.names)
    
    # Convert FITS table to pandas DataFrame
    xmm_data = pd.DataFrame(hdul[1].data)
    
    # Select only the columns we need for AGN classification
    needed_columns = ['RA', 'DEC', 'EP_8_FLUX', 'EP_8_FLUX_ERR', 'EP_1_FLUX', 'EP_1_FLUX_ERR']
    xmm_data = xmm_data[needed_columns]

# Print XMM data info
print("\nXMM data shape:", xmm_data.shape)
print("\nXMM first few rows:")
print(xmm_data.head())

# Clean SDSS data
sdss_data = sdss_data.drop_duplicates(subset=['ra', 'dec'])
sdss_data = sdss_data.dropna(subset=['ra', 'dec', 'redshift'])

# Clean XMM-Newton data
xmm_data = xmm_data.dropna(subset=['RA', 'DEC'])

print("\nAfter cleaning:")
print("SDSS data shape:", sdss_data.shape)
print("XMM data shape:", xmm_data.shape)

# Convert coordinates to SkyCoord objects
sdss_coords = SkyCoord(ra=sdss_data['ra'], dec=sdss_data['dec'], unit=(u.degree, u.degree))
xmm_coords = SkyCoord(ra=xmm_data['RA'], dec=xmm_data['DEC'], unit=(u.degree, u.degree))

# Perform cross-matching
max_sep = 3 * u.arcsec
idx, d2d, _ = sdss_coords.match_to_catalog_sky(xmm_coords)
matches = d2d < max_sep

print("\nNumber of matches found:", np.sum(matches))

# Only proceed if matches were found
if np.sum(matches) > 0:
    # Merge datasets
    merged_data = pd.concat([
        sdss_data[matches].reset_index(drop=True),
        xmm_data.iloc[idx[matches]].reset_index(drop=True)
    ], axis=1)

    print("\nMerged data shape:", merged_data.shape)

    # Create visualizations and continue with the rest of the processing
    plt.figure(figsize=(12, 6))
    sns.heatmap(merged_data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Before Imputation')
    plt.tight_layout()
    plt.savefig('missing_values_before.png')
    plt.close()

    # Merge datasets
    merged_data = pd.concat([
        sdss_data[matches].reset_index(drop=True),
        xmm_data.iloc[idx[matches]].reset_index(drop=True)
    ], axis=1)

    # Visualize missing values before imputation
    plt.figure(figsize=(12, 6))
    sns.heatmap(merged_data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Before Imputation')
    plt.tight_layout()
    plt.savefig('missing_values_before.png')
    plt.close()

    # Handle missing values
    numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    merged_data[numeric_columns] = imputer.fit_transform(merged_data[numeric_columns])

    # Visualize missing values after imputation
    plt.figure(figsize=(12, 6))
    sns.heatmap(merged_data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values After Imputation')
    plt.tight_layout()
    plt.savefig('missing_values_after.png')
    plt.close()

    # Feature engineering
    merged_data['optical_xray_ratio'] = merged_data['r'] / merged_data['EP_8_FLUX']
    merged_data['xray_hardness_ratio'] = (merged_data['EP_8_FLUX'] - merged_data['EP_1_FLUX']) / (merged_data['EP_8_FLUX'] + merged_data['EP_1_FLUX'])

    # Normalize features
    scaler = StandardScaler()
    merged_data[numeric_columns] = scaler.fit_transform(merged_data[numeric_columns])

    # Visualize feature distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(numeric_columns[:9]):  # Plot first 9 features
        plt.subplot(3, 3, i+1)
        sns.histplot(merged_data[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(merged_data[numeric_columns].corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Save preprocessed data
    merged_data.to_csv("D:\\preprocessed_agn_data.csv", index=False)

    print("Data preprocessing completed. Preprocessed data saved to D:\\preprocessed_agn_data.csv")
    print("Visualization images saved: missing_values_before.png, missing_values_after.png, feature_distributions.png, correlation_heatmap.png")
    
else:
    print("\nNo matches found between SDSS and XMM-Newton data!")
    print("Please check the following:")
    print("1. Coordinate systems and units")
    print("2. Search radius (currently 3 arcsec)")
    print("3. Data quality and coverage overlap")

