import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_dbscan_and_clean_data(data, eps, min_samples, metric, algorithm, leaf_size, dataset_name,  plot=False, **kwargs):
    """
    Creates a DBSCAN model with user-specified parameters, removes detected outliers,
    and returns the cleaned data with a new column 'cluster' for cluster labels.
    Additionally, saves outliers and optionally plots the data.

    Parameters:
        data (pd.DataFrame or np.ndarray): The input data.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        metric (str): The distance metric to use ('euclidean', 'manhattan', 'mahalanobis', etc.).
        algorithm (str): The algorithm to be used by the NearestNeighbors module ('auto', 'ball_tree', 'kd_tree', 'brute').
        leaf_size (int): Leaf size passed to BallTree or cKDTree.
        p (float): The power of the Minkowski metric to be used (e.g., 1 for Manhattan, 2 for Euclidean).
        plot (bool): If True, plots the data points (blue for non-outliers, red for outliers).

    Returns:
        pd.DataFrame: The cleaned data with outliers removed and a 'cluster' column added.
    """
    # Ensure the input is a DataFrame
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])

    # Standardize the data
    standardized_data = StandardScaler().fit_transform(data)

    # Create the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size,  **kwargs)
    
    # Fit the model and predict labels
    labels = dbscan.fit_predict(standardized_data)
    
    # Add the cluster labels as a new column to the original dataframe
    data['cluster'] = labels
    
    # Identify the non-outlier data (labels != -1)
    clean_data = data[data['cluster'] != -1].reset_index(drop=True)
    
    
    
    # Print statistics
    print(f"Percentage of outliers removed: {sum(labels == -1) / len(data) * 100:.2f}%")
    print(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    
    # Optional plotting
    if plot:
        # Use PCA for dimensionality reduction if data has more than 2 features
        features = data.columns[:-1]  # Exclude 'cluster'
        if len(features) > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(standardized_data)
        else:
            reduced_data = data[features].values
        
        # Plot non-outliers (blue) and outliers (red)
        plt.figure(figsize=(10, 7))
        plt.scatter(reduced_data[labels != -1, 0], reduced_data[labels != -1, 1], 
                    color='blue', label='Non-Outliers', alpha=0.5)
        plt.scatter(reduced_data[labels == -1, 0], reduced_data[labels == -1, 1], 
                    color='red', label='Outliers', alpha=0.5)
        plt.title('DBSCAN: Outliers vs Non-Outliers')
        plt.xlabel('Component 1' if len(features) > 2 else features[0])
        plt.ylabel('Component 2' if len(features) > 2 else features[1])
        plt.legend()
        plt.show()
    
    # Save outliers (labels == -1) ignoring the 'cluster' column
    outliers = data[data['cluster'] == -1].drop(columns=['cluster']).reset_index(drop=True)
    # outliers.drop(['cluster'], axis=1, inplace=True)
    outliers.to_csv(f'../data/{dataset_name}_3_outliers_by_DBSCAN.csv', index=False)

    # Save non-outlier data
    clean_data.to_csv(f'../data/{dataset_name}_2_cleaned_by_DBSCAN.csv', index=False)
    
    return clean_data

# Example usage:
# dataset_name = 'blobs'
# data, _ = dataset_options[dataset_name]()
# cleaned_data = create_dbscan_and_clean_data(data, eps=0.5, min_samples=5, metric='euclidean', 
#                                             algorithm='auto', leaf_size=30, p=2, plot=True)