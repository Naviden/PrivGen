import sys
import os
from sklearn.impute import SimpleImputer

# Adds the parent directory to sys.path
sys.path.append(os.path.abspath('..'))
import pandas as pd
from utils import (data_encoder, hyper_param_opt,
                   outlier_detection_by_clust,
                   weighted_distance_calculation,
                   outlier_detection_by_dist,
                   final_plot, multiproc_DBSCAN)
import pickle


data = pd.read_csv('../datasets/healthinsurance.csv')

cols = data.columns
imp = SimpleImputer(strategy="most_frequent")
data = pd.DataFrame(imp.fit_transform(data))
data.columns = cols


dataset_name= 'health'
encoded_data = data_encoder.ordinal_encode_categorical(
        data, dataset_name, save=True)

multiproc_DBSCAN.hyper_param_search(
    encoded_data,
    percentage_outliers=0.1,
    dataset_name=dataset_name,
    top_n=10,
    eps_values=[0.7, 0.8, 0.9, 1., 1.5, 2],
    min_samples_values=range(30, 50),
    save=True)