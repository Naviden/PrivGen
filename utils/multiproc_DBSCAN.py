import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from multiprocessing import Pool, cpu_count

def multiprocess_dbscan(args):
    data, eps, min_samples, metric, algorithm, leaf_size, p = args
    
    if metric == 'mahalanobis':
        try:
            distance_matrix = pairwise_distances(data, metric='mahalanobis')
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = dbscan.fit_predict(distance_matrix)
        except:
            return None
    else:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                        algorithm=algorithm, leaf_size=leaf_size, p=p)
        labels = dbscan.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    silhouette = -1
    
    if 1 < len(set(labels)) < len(data):
        try:
            silhouette = silhouette_score(data, labels)
        except ValueError:
            pass
    
    return {
        'num_clusters': n_clusters,
        'percentage_outliers': n_outliers / len(data),
        'eps': eps,
        'min_samples': min_samples,
        'metric': metric,
        'algorithm': algorithm,
        'leaf_size': leaf_size,
        'p': p,
        'silhouette_score': silhouette
    }

def hyper_param_search(data, percentage_outliers, dataset_name, num_clusters=None, top_n=5, plot=False,
                       eps_values=np.linspace(0.1, 1.0, 10), min_samples_values=range(3, 10),
                       metrics=['euclidean', 'manhattan', 'mahalanobis'],
                       algorithms=['auto', 'ball_tree', 'kd_tree', 'brute'],
                       leaf_sizes=range(10, 60, 10),
                       p_values=[None, 1, 2],
                       save=True):
    
    data = StandardScaler().fit_transform(data)
    param_combinations = list(itertools.product(eps_values, min_samples_values, metrics, algorithms, leaf_sizes, p_values))
    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(multiprocess_dbscan, [(data, *params) for params in param_combinations]),
                            total=len(param_combinations), desc="Processing DBSCAN parameters"))
    
    results = [res for res in results if res is not None]
    all_results = pd.DataFrame(results)
    
    if save:
        all_results.to_csv(f'../data/{dataset_name}_grid_search_results.csv', index=False)
    
    top_results = find_top_records(
        all_results, percentage_outliers=percentage_outliers, num_clusters=num_clusters, top_n=top_n)
    
    return top_results
