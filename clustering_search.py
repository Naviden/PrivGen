import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification, make_gaussian_quantiles
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import itertools

# Custom metric for Mahalanobis
def mahalanobis_distance(X):
    cov = EmpiricalCovariance().fit(X)
    inv_cov = cov.precision_
    return lambda u, v: mahalanobis(u, v, inv_cov)

def run_clustering_and_plot():
    # Dataset options
    dataset_options = {
        'blobs': lambda: make_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=42),
        'moons': lambda: make_moons(n_samples=500, noise=0.05, random_state=42),
        'circles': lambda: make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42),
        'classification': lambda: make_classification(n_samples=500, n_features=2, n_informative=2, n_clusters_per_class=1, random_state=42),
        'gaussian_quantiles': lambda: make_gaussian_quantiles(n_samples=500, n_features=2, random_state=42)
    }

    # Choose dataset
    dataset_name = 'blobs'  # Change this to use a different dataset
    data, labels_true = dataset_options[dataset_name]()

    # Standardize dataset
    data = StandardScaler().fit_transform(data)

    # Hyperparameter grid
    eps_values = np.linspace(0.1, 1.0, 10)
    min_samples_values = range(3, 10)
    metrics = ['euclidean', 'manhattan', mahalanobis_distance(data)]

    # Results storage
    results = []

    # Perform grid search
    for eps, min_samples, metric in itertools.product(eps_values, min_samples_values, metrics):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric if not callable(metric) else 'precomputed').fit(data)
        labels = dbscan.fit_predict(data)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)

        silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else -1

        results.append({
            'num_clusters': n_clusters,
            'num_outliers': n_outliers,
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric if isinstance(metric, str) else 'mahalanobis',
            'silhouette_score': silhouette
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plotting
    fig, axes = plt.subplots(len(min_samples_values), len(eps_values), figsize=(20, 15))
    fig.tight_layout(pad=3.0)

    for (eps, min_samples, metric), ax in zip(itertools.product(eps_values, min_samples_values, metrics), axes.flat):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(data)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)

        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
        ax.set_title(
            f'eps={eps:.2f}, min_samples={min_samples}, metric={"mahalanobis" if callable(metric) else metric}'
            f'\nClusters: {n_clusters}, Outliers: {n_outliers}',
            fontsize=8
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig('clusters.pdf', dpi=600)

    return results_df