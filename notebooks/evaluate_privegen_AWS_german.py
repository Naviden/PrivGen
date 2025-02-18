from synthcity.metrics import Metrics
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import pandas as pd
import time
from memory_profiler import memory_usage
from glob import glob
from sklearn.preprocessing import OrdinalEncoder
import pickle
                   
main_path = "./"


def compare_results(results1: pd.DataFrame, results2: pd.DataFrame):
    """
    used in eval_5 function
    Compares two results tables and determines which results are better for each metric.
    Handles .gt and .syn metrics by focusing only on .syn for comparisons. Adds actual values.

    Args:
        results1 (pd.DataFrame): The first results table.
        results2 (pd.DataFrame): The second results table.

    Returns:
        pd.DataFrame: A DataFrame summarizing which results are better for each .syn metric.
    """
    comparison = []

    for metric in results1.index:
        # Skip ground truth (.gt) metrics
        if metric.endswith(".gt"):
            continue

        # Ensure the corresponding .gt scores are equal before comparing .syn metrics
        if metric.endswith(".syn"):
            corresponding_gt_metric = metric.replace(".syn", ".gt")
            if (results1.loc[corresponding_gt_metric, "mean"] != results2.loc[corresponding_gt_metric, "mean"]):
                comparison.append([metric, "Error", "Ground truth values differ", None, None])
                continue

        # Determine comparison direction
        direction = results1.loc[metric, "direction"]
        value1 = results1.loc[metric, "mean"]
        value2 = results2.loc[metric, "mean"]

        if direction == "minimize":
            diff = value2 - value1
            better = "results 1" if diff > 0 else "results 2"
        elif direction == "maximize":
            diff = value1 - value2
            better = "results 1" if diff > 0 else "results 2"
        else:
            better = "N/A"

        magnitude = "marginally" if abs(diff) < 0.05 else "by far"
        comparison.append([metric, better, magnitude, value1, value2])

    return pd.DataFrame(comparison, columns=["Metric", "Better Results", "Magnitude", "orig", "privgen"])

def eval_5(base, source, target_column, sensetive_columns, data_name,
            file_name=None, task='regression', synthesizer='tvae',
           metrics=['sanity', 'stats', 'privacy']):

    """
    creates a csv file that compares the synthetic data generated from original data (base) and original data+Privegen (source)
    It applies all available synthsizers on two versions of data and then evaluates them
    """
    available_metrics = {
    'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
    'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
    'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
    'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
    'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
    }

    available_metrics = {
        'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability'],
        'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision'], # missing: Frechet inception dictance
        'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
        }
    existing = glob(f'/home/ec2-user/PrivGen/intermed_results/{data_name}/*.csv')
    if f'/home/ec2-user/PrivGen/intermed_results/{data_name}/{synthesizer}_orig.csv' not in existing:
        needed_metrics = {k:available_metrics[k] for k in metrics}

        results = []
        for i, src in enumerate([base, source]):

            X = src.copy()
            plugin = Plugins().get(synthesizer)
            loader = GenericDataLoader(X, target_column=target_column,
                                        sensitive_columns=sensetive_columns)
            plugin.fit(loader)
            synth = plugin.generate(len(X), random_state=42).dataframe()


            score = Metrics.evaluate(
            X_gt=base,
            X_syn=synth,
            metrics=needed_metrics,
            task_type=task,
            random_state=42
            )
            if i == 0:
                score['privgen_used'] = [False for i in range(len(score))]
                name = 'orig'
            else:
                score['privgen_used'] = [True for i in range(len(score))]
                name = 'prigen'
            score = score.reset_index()
            score.to_csv(f'../intermed_results/{data_name}/{synthesizer}_{name}.csv', index=False)

            results.append(score)
        else:
            print('skipping...')
#   comparison = compare_results(*results)

#   # saving results
#   if not file_name:
#     file_name = f'{data_name}_{synthesizer}'
#   comparison.to_csv(f'{save_path}/{file_name}.csv', index=False)
#   return comparison



plugins = Plugins()
plugin_names = plugins.list() 

# ----------------------------------------
# dataset info
data_name = 'german' 
dataset_name= 'german'

target_column = "class"
sensetive_columns = ['Attribute 10']
# ----------------------------------------


def ordinal_encode_categorical(data, dataset_name, save=True):
    """
    Encodes categorical columns using OrdinalEncoder and saves mappings for reversibility.

    Parameters:
    data (pd.DataFrame): Input DataFrame with categorical columns to encode.
    pickle_path (str): File path to save mappings and column mappings (default is 'mappings.pkl').

    Returns:
    pd.DataFrame: Encoded DataFrame with numerical values replacing categorical columns.
    """

    pickle_path = f'{dataset_name}_mappings.pkl'
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    mappings = {}
    encoder = OrdinalEncoder()
    
    # Apply encoding only to categorical columns
    encoded_data = data.copy()
    encoded_data[categorical_columns] = encoder.fit_transform(data[categorical_columns])
    
    # Store mappings for each column
    for i, col in enumerate(categorical_columns):
        mappings[col] = {
            'categories': encoder.categories_[i]
        }
    
    # Save mappings to a pickle file
    with open(pickle_path, 'wb') as f:
        pickle.dump(mappings, f)
    if save:
        encoded_data.to_csv(f'../data/{dataset_name}_1_encoded_data.csv', index=False)
    return encoded_data

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
data = X
data['class'] = y
base = data
# base = ordinal_encode_categorical(
#         base, dataset_name, save=False)

source = pd.read_csv(f'../data/{data_name}_8_decoded_data.csv')
# source = ordinal_encode_categorical(
#         source, dataset_name, save=False)

#inputation
from sklearn.impute import SimpleImputer

cols = base.columns
imp = SimpleImputer(strategy="most_frequent")
base = pd.DataFrame(imp.fit_transform(base))
base.columns = cols



# all synthesizers
plugins = Plugins()
plugin_names = plugins.list()

# fuck privbayes, great
plugin_names = ['tvae', 'dpgan', 'adsgan', 'pategan', 'marginal_distributions',
 'dummy_sampler', 'nflow', 'arf',   'ctgan', 
  'decaf',  'ddpm', 'uniform_sampler',  'rtvae']

for synthesizer in plugin_names:
  # we don't need survival data AND "aim" model gives errors and we dont want to re-run what we have already done, so...
#   if f'{data_name}_{synthesizer}' not in existing and 'survival' not in synthesizer and synthesizer != 'aim':

    print(f'Running {synthesizer}. . . ')
    # try:
    start_time = time.time()
    # Monitor memory usage during execution
    eval_5(base, source, target_column, sensetive_columns, data_name, synthesizer=synthesizer)

    end_time = time.time()
    execution_time = end_time - start_time



    # Performance log
    log_line = (
        f"data_name: {data_name}, "
        f"synthesizer: {synthesizer}, "
        f"execution_time: {execution_time:.6f} seconds, "    )

    with open('../results/performance_log_2.txt', "a") as log_file:
        log_file.write(log_line)
    # except Exception as e:
    #   print('='*80)
    #   print(f"An error  with {synthesizer}: {e}")
    #   print('='*80)