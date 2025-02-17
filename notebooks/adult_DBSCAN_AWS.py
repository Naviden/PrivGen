import sys
import os
# Adds the parent directory to sys.path
sys.path.append(os.path.abspath('..'))
import pandas as pd
from utils import (data_encoder, hyper_param_opt,
                   outlier_detection_by_clust,
                   weighted_distance_calculation,
                   outlier_detection_by_dist,
                   final_plot, multiproc_DBSCAN)
import pickle
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

data = X
data['income'] = y
from sklearn.impute import SimpleImputer

cols = data.columns
imp = SimpleImputer(strategy="most_frequent")
data = pd.DataFrame(imp.fit_transform(data))
data.columns = cols

HALF_SAMPLE_SIZE = 1000

data.income.replace('<=50K', '<=50K.', inplace=True )
data.income.replace('>50K', '>50K.', inplace=True )
data1 = data[data['income'] == '<=50K.'].sample(HALF_SAMPLE_SIZE, random_state=42)
data2 = data[data['income'] == '>50K.'].sample(HALF_SAMPLE_SIZE, random_state=42)
data = pd.concat([data1, data2])
data.reset_index(drop=True, inplace=True)

data['age'] = data['age'].astype(int)
data['fnlwgt'] = data['fnlwgt'].astype(int)
data['capital-gain'] = data['capital-gain'].astype(int)
data['capital-loss'] = data['capital-loss'].astype(int)
data['hours-per-week'] = data['hours-per-week'].astype(int)

dataset_name= 'adult'
encoded_data = data_encoder.one_hot_encode_categorical(
        data, dataset_name, save=True)

multiproc_DBSCAN.hyper_param_search(
    encoded_data,
    percentage_outliers=0.1,
    dataset_name=dataset_name,
    top_n=10,
    eps_values=[0.7, 0.8, 0.9, 1., 1.5, 2],
    min_samples_values=range(30, 50),
    save=True)