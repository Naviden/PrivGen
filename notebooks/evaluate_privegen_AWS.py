from synthcity.metrics import Metrics
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import pandas as pd
import time
from memory_profiler import memory_usage
from glob import glob

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
           save_path, file_name=None, task='regression', synthesizer='tvae',
           metrics=['sanity', 'stats', 'performance', 'detection', 'privacy']):

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

  needed_metrics = {k:available_metrics[k] for k in metrics}

  results = []
  for src in [base, source]:

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

    results.append(score)

  comparison = compare_results(*results)

  # saving results
  if not file_name:
    file_name = f'{data_name}_{synthesizer}'
  comparison.to_csv(f'{save_path}/{file_name}.csv', index=False)
  return comparison



plugins = Plugins()
plugin_names = plugins.list() 

def download_dataset(dataset_name):
    # Map dataset names to UCI ML repository IDs
    dataset_ids = {
    "iris": {"id": 53, "target_column": "target", "sensitive_columns": ["sepal width"]},
    "wine": {"id": 109, "target_column": "target", "sensitive_columns": ["Alcohol", "Malicacid","Ash","Alcalinity_of_ash"]},
    "abalone": {"id": 1, "target_column": "target", "sensitive_columns": ["Sex"]},
    "heart disease": {"id": 45, "target_column": "target", "sensitive_columns": ["age", "sex"]},
    "adult": {"id": 2, "target_column": "target", "sensitive_columns": ["age", "race", "sex"]},
    "car evaluation": {"id": 19, "target_column": "target", "sensitive_columns": ["buying"]},
    "automobile": {"id": 10, "target_column": "target", "sensitive_columns": ["price"]},
    "mushroom": {"id": 73, "target_column": "target", "sensitive_columns": ["odor", "bruises"]},
    "german credit": {"id": 144, "target_column": "target", "sensitive_columns": ["Attribute1","Attribute2"]},
    "dry bean": {"id": 602, "target_column": "target", "sensitive_columns": ["Area","Perimeter","MajorAxisLength"]},
    "bike sharing": {"id": 275, "target_column": "target", "sensitive_columns": []},
    "auto_mpg": {"id": 9, "target_column": "target", "sensitive_columns": ["origin"]},
    "RT-IoT2022": {"id": 942, "target_column": "target", "sensitive_columns": []},
    "EEG Eye State": {"id": 264, "target_column": "target", "sensitive_columns": []},
    "Metro": {"id": 492, "target_column": "target", "sensitive_columns": []}
    }
    
    
    # Fetch the dataset ID
    dataset_id = dataset_ids[dataset_name]['id']
    
    # Fetch the dataset
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
    y = dataset.data.targets
    X['target'] = y
    
    return X, dataset_ids[dataset_name]['target_column'], dataset_ids[dataset_name]['sensitive_columns']



base = pd.read_csv('./datasets/cervical-cancer_csv.csv')
source = pd.read_csv('./data/cervical_8_decoded_data.csv')
target_column = "Biopsy"
sensetive_columns = []



files = glob('./results/*.csv') # where we save the results - current path works is Ok for GDrive
existing = [e.split('/')[-1].split('.')[0] for e in files]

# here we can use abalone dataset because we previously applied privgen to it (privgen_example.ipynb) and the required artifacts are already in data directory
data_name = 'cervical' 

# all synthesizers
plugins = Plugins()
plugin_names = plugins.list()

for synthesizer in plugin_names:
  # we don't need survival data AND "aim" model gives errors and we dont want to re-run what we have already done, so...
  if f'{data_name}_{synthesizer}' not in existing and 'survival' not in synthesizer and synthesizer != 'aim':

    print(f'Running {synthesizer}. . . ')
    try:
      start_time = time.time()
      # Monitor memory usage during execution
      start_memory, peak_memory = memory_usage((eval_5(base, source, target_column, sensetive_columns, data_name,
                save_path='./results', synthesizer=synthesizer)), retval=False, interval=0.1, timeout=None, max_usage=True)
      end_memory = memory_usage(-1, retval=False)[0]  # Memory after function ends

      end_time = time.time()
      execution_time = end_time - start_time

      # Calculate memory differences
      memory_diff = end_memory - start_memory

      # Performance log
      log_line = (
          f"data_name: {data_name}, "
          f"synthesizer: {synthesizer}, "
          f"execution_time: {execution_time:.6f} seconds, "
          f"Peak Memory Usage: {peak_memory:.2f} MB, "
          f"memory_diff_MB: {memory_diff:.2f}\n"
      )

      with open('./results/performance_log.txt', "a") as log_file:
          log_file.write(log_line)
    except Exception as e:
      print('='*80)
      print(f"An error  with {synthesizer}: {e}")
      print('='*80)