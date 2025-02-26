import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def geometric_median(X, eps=1e-5):
    y = np.mean(X, axis=0)
    while True:
        D = np.linalg.norm(X - y, axis=1)
        nonzeros = (D != 0)
        D_inv = 1 / D[nonzeros]
        W = D_inv / D_inv.sum()
        T = (W[:, np.newaxis] * X[nonzeros]).sum(axis=0)
        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * np.sum(D_inv)
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y
        if np.linalg.norm(y - y1) < eps:
            return y1
        y = y1

def cluster_and_analyze(data, dataset_name, feature_weights=None):
    """
    Adds a 'distance' column to the data, representing the weighted distance 
    of each point from its cluster's geometric median.

    Parameters:
        data (pd.DataFrame): Input data with a 'cluster' column.
        dataset_name (str): Name of the dataset for saving results.
        feature_weights (dict, optional): Dictionary of feature weights {feature_name: weight}.
                                          If None, all features have equal weight.

    Returns:
        pd.DataFrame: Data with an additional 'distance' column.
    """
    # Ensure input is a DataFrame and extract numeric columns
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    numeric_columns = [col for col in data.columns if col != 'cluster']
    
    # Set equal weights if feature_weights is not provided
    if feature_weights is None:
        weights = np.ones(len(numeric_columns)) / len(numeric_columns)
    else:
        raw_weights = np.array([feature_weights.get(col, 1.0) for col in numeric_columns])
        weights = raw_weights / np.sum(raw_weights)  # Normalize weights to sum to 1

    result_frames = []
    
    for cluster_label in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster_label].drop(columns=['cluster'])
        median = geometric_median(cluster_data.values)
        
        # Calculate weighted distance for each point
        def weighted_distance(row):
            return np.sqrt(np.sum(weights * (row.values - median) ** 2))

        distances = cluster_data.apply(weighted_distance, axis=1)
        
        # Add distances as a new column to the original cluster data
        cluster_data_with_distances = data[data['cluster'] == cluster_label].copy()
        cluster_data_with_distances['distance'] = distances
        
        result_frames.append(cluster_data_with_distances)
    
    final_df = pd.concat(result_frames).reset_index(drop=True)
    final_df.to_csv(f'../data/{dataset_name}_4_weighted_distances.csv', index=False)
    
    return final_df

def calculate_plot_distance(dataset_name):

    data = pd.read_csv(f'../data/{dataset_name}_2_cleaned_by_DBSCAN.csv')
    data = cluster_and_analyze(data, dataset_name)

    for cluster_label in data['cluster'].unique():
        cluster_distances = data[data['cluster'] == cluster_label]['distance']
        plt.figure()
        plt.hist(cluster_distances, bins=30, density=True, alpha=0.6, edgecolor='k')
        plt.title(f'Cluster {cluster_label} - Distance Density Histogram')
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.show()

# Example usage:
# data = pd.DataFrame(...)  # Your dataset
# eps = 0.5
# min_samples = 5
# metric = 'euclidean'
# final_df = cluster_and_analyze(data, eps, min_samples, metric)
# plot_density_histograms(final_df)