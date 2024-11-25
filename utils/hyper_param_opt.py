import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification, make_gaussian_quantiles
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import itertools
from tqdm import tqdm

# Custom metric for Mahalanobis
def mahalanobis_distance(X):
    cov = EmpiricalCovariance().fit(X)
    inv_cov = cov.precision_
    return lambda u, v: mahalanobis(u, v, inv_cov)



def find_top_records(dataframe, percentage_outliers, num_clusters=None, top_n=5):
    """
    Filters the dataframe to return top N rows based on the closest percentage_outliers.
    If num_clusters is provided and exact match isn't available, it finds the closest num_clusters value.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        percentage_outliers (float): The target percentage of outliers.
        num_clusters (int, optional): The specified number of clusters. Defaults to None.
        top_n (int): Number of top records to return (default is 5).

    Returns:
        pd.DataFrame: Filtered dataframe with the top N closest rows.
    """
    if num_clusters is not None:
        # Check if exact num_clusters exists
        if num_clusters in dataframe['num_clusters'].values:
            filtered = dataframe[dataframe['num_clusters'] == num_clusters]
        else:
            # Find the closest num_clusters value
            closest_cluster = dataframe['num_clusters'].sub(num_clusters).abs().idxmin()
            closest_value = dataframe.loc[closest_cluster, 'num_clusters']
            filtered = dataframe[dataframe['num_clusters'] == closest_value]
    else:
        filtered = dataframe
    
    # Calculate the absolute difference in 'percentage_outliers'
    filtered['outlier_difference'] = (filtered['percentage_outliers'] - percentage_outliers).abs()
    
    # Sort by the absolute difference and return the top N rows
    top_records = filtered.sort_values('outlier_difference').head(top_n)
    
    return top_records.drop(columns=['outlier_difference'])



def hyper_param_search(data, percentage_outliers, dataset_name, num_clusters=None, top_n=5, plot=False,
                       eps_values=np.linspace(0.1, 1.0, 10), min_samples_values=range(3, 10),
                       metrics=['euclidean', 'manhattan', 'mahalanobis'], 
                       algorithms=['auto', 'ball_tree', 'kd_tree', 'brute'], 
                       leaf_sizes=range(10, 60, 10), 
                       p_values=[None, 1, 2],
                       
                       save=True):
    """
    Runs DBSCAN clustering with various parameters and optionally plots the results.

    Parameters:
        data (ndarray): Input data for clustering.
        percentage_outliers (float): Target percentage of outliers.
        num_clusters (int, optional): Target number of clusters. Defaults to None.
        top_n (int): Number of top records to return. Defaults to 5.
        plot (bool): Whether to plot the clustering results. Defaults to False.
        eps_values (iterable): Range of epsilon values for DBSCAN. Defaults to np.linspace(0.1, 2.0, 10).
        min_samples_values (iterable): Range of min_samples for DBSCAN. Defaults to range(3, 30).
        metrics (list): List of distance metrics. Defaults to ['euclidean', 'manhattan', 'mahalanobis'].
        algorithms (list): List of algorithms for DBSCAN. Defaults to ['auto', 'ball_tree', 'kd_tree', 'brute'].
        leaf_sizes (iterable): Range of leaf_size for DBSCAN. Defaults to range(10, 110, 10).
        p_values (list): List of Minkowski metric powers. Defaults to [None, 1, 2].
        save: if True, the grid search results are saved

    Returns:
        pd.DataFrame: Dataframe containing the top N clustering results.
    """
    comm = 0  # used for error printing
    data = StandardScaler().fit_transform(data)

    results = []

    if plot:
        fig, axes = plt.subplots(len(min_samples_values), len(eps_values), figsize=(20, 15))
        fig.tight_layout(pad=3.0)

    for i, ((eps, min_samples, metric, algorithm, leaf_size, p), ax) in enumerate(
        zip(
            tqdm(itertools.product(eps_values, min_samples_values, metrics, algorithms, leaf_sizes, p_values),
                 total=len(eps_values) * len(min_samples_values) * len(metrics) * len(algorithms) * len(leaf_sizes) * len(p_values),
                 desc="Processing DBSCAN parameters"), 
            axes.flat if plot else itertools.repeat(None)
        )
    ):
        if metric == 'mahalanobis':
            try:
                # Precompute the pairwise Mahalanobis distances
                distance_matrix = pairwise_distances(data, metric='mahalanobis')
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                labels = dbscan.fit_predict(distance_matrix)
            except:
                if comm == 0:
                    print('Mahalanobis distance cannot be used...')
                comm = 1
        else:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                            algorithm=algorithm, leaf_size=leaf_size, p=p)
            labels = dbscan.fit_predict(data)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)

        if plot:
            ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
            ax.set_title(
                f'eps={eps:.2f}, min_samples={min_samples}, metric={metric}, alg={algorithm}, leaf={leaf_size}, p={p}'
                f'\nClusters: {n_clusters}, Outliers: {n_outliers}',
                fontsize=8
            )
            ax.set_xticks([])
            ax.set_yticks([])

        silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else -1

        results.append({
            'num_clusters': n_clusters,
            'percentage_outliers': n_outliers / len(data),
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'p': p,
            'silhouette_score': silhouette
        })

    if plot:
        plt.savefig('clusters.pdf', dpi=600)
    
    all_results = pd.DataFrame(results)
    if save:
        all_results.to_csv(f'../data/{dataset_name}_grid_search_results.csv', index=False)
    top_results = find_top_records(all_results, percentage_outliers=percentage_outliers, num_clusters=num_clusters, top_n=top_n)
    
    return top_results