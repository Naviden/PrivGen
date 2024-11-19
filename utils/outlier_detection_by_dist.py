import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def filter_and_save_by_cluster(path, cluster_thresholds, plot=False):
    """
    Filters a dataframe based on cluster thresholds, saves the filtered dataframe to one CSV,
    and saves the outliers (excluding cluster and distance columns) to another CSV. Optionally plots the data.

    Parameters:
    path (str): The path to the input CSV file.
    cluster_thresholds (dict): A dictionary where keys are cluster names as strings, 
                               and values are the threshold for the distance.
    plot (bool): If True, plots the data points (blue for non-outliers, red for outliers).
    """
    # Load the dataframe
    dataframe = pd.read_csv(path)
    
    # Ensure 'cluster' column is string type
    dataframe['cluster'] = dataframe['cluster'].astype(str)
    
    # Filter the dataframe for each cluster based on the provided thresholds
    filtered_dfs = []
    outlier_dfs = []
    
    for cluster, threshold in cluster_thresholds.items():
        cluster_data = dataframe[dataframe['cluster'] == cluster]
        filtered_cluster = cluster_data[cluster_data['distance'] <= threshold]
        outliers_cluster = cluster_data[cluster_data['distance'] > threshold]
        
        filtered_dfs.append(filtered_cluster)
        outlier_dfs.append(outliers_cluster.drop(['cluster', 'distance'], axis=1))
    
    # Concatenate filtered data and save
    filtered_dataframe = pd.concat(filtered_dfs)
    filtered_dataframe.to_csv('./data/final_cleaned.csv', index=False)
    print("Filtered dataframe has been saved to './data/final_cleaned.csv'")
    
    # Concatenate outliers and save
    outliers_dataframe = pd.concat(outlier_dfs)
    # outliers_dataframe.drop(['cluster', 'distance'], axis=1, inplace=True)
    outliers_dataframe.to_csv('./data/outliers_by_distance.csv', index=False)
    print("Outliers have been saved to './data/outliers.csv'")
    
    # Plot the data if requested
    if plot:
        # Combine filtered and outliers for consistent plotting
        filtered_dataframe['is_outlier'] = False
        outliers_dataframe['is_outlier'] = True
        combined = pd.concat([filtered_dataframe, outliers_dataframe])
        
        # Apply PCA if more than 2 dimensions
        features = [col for col in dataframe.columns if col not in ['cluster', 'distance']]
        if len(features) > 2:
            pca = PCA(n_components=2)
            combined[['PCA1', 'PCA2']] = pca.fit_transform(combined[features])
            x, y = 'PCA1', 'PCA2'
        else:
            x, y = features[0], features[1]
        
        # Plot
        plt.figure(figsize=(10, 7))
        plt.scatter(combined[combined['is_outlier'] == False][x], 
                    combined[combined['is_outlier'] == False][y], 
                    color='blue', label='Non-Outliers', alpha=0.5)
        plt.scatter(combined[combined['is_outlier'] == True][x], 
                    combined[combined['is_outlier'] == True][y], 
                    color='red', label='Outliers', alpha=0.5)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title('Outliers vs Non-Outliers')
        plt.legend()
        plt.show()