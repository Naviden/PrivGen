import pandas as pd
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_privacy import (
    DeltaPresence,
    kAnonymization,
    kMap,
    lDiversityDistinct,
    IdentifiabilityScore
 )
# from synthcity.metrics.eval_sanity import (
#     DataMismatchScore,
#     CommonRowsProportion,
#     NearestSynNeighborDistance,
#     CloseValuesProbability,
#     DistantValuesProbability
# )
# from synthcity.metrics.eval_stats import (
#     JensenShannonDist,
#     ChiSquaredTest,
#     FeatureCorrelation,
#     InverseKLDivergence,
#     KSTest,
#     MaximumMeanDiscrepancy,
#     WassersteinDistance,
#     PRDCScore,
#     AlphaPrecision,
#     SurvivalKMDistance
# )
# from synthcity.metrics.eval_performance import (
#     PerformanceLinear,
#     PerformanceMLP,
#     PerformanceXGB,
#     FeatureRankDistance
# )
# from synthcity.metrics.eval_detection import (
#     DetectionXGB,
#     DetectionMLP,
#     DetectionGMM,
#     DetectionLinear
# )

def evaluate_all_metrics(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, sensitive_columns: list, target_column: str = None):
    """
    Evaluate all specified metrics for real and synthetic datasets.

    Parameters:
    - real_data: DataFrame containing the real dataset.
    - synthetic_data: DataFrame containing the synthetic dataset.
    - sensitive_columns: List of column names considered as sensitive.
    - target_column: Optional column name for performance metrics.

    Returns:
    - Dictionary containing the evaluated metrics grouped by category.
    """
    # Load data into GenericDataLoader
    real_loader = GenericDataLoader(real_data, sensitive_columns=sensitive_columns)
    synthetic_loader = GenericDataLoader(synthetic_data, sensitive_columns=sensitive_columns)

    # Initialize results
    results = {}

    # Privacy Metrics
    results['privacy'] = {
        'delta_presence': DeltaPresence().evaluate(real_loader, synthetic_loader),
        'k_anonymity': kAnonymization().evaluate(real_loader, synthetic_loader),
        'k_map': kMap().evaluate(real_loader, synthetic_loader),
        'l_diversity': lDiversityDistinct().evaluate(real_loader, synthetic_loader),
        # 'identifiability_score': IdentifiabilityScore().evaluate(real_loader, synthetic_loader)
    }

    # # Sanity Metrics
    # results['sanity'] = {
    #     'data_mismatch': DataMismatchScore().evaluate(real_loader, synthetic_loader),
    #     'common_rows_proportion': CommonRowsProportion().evaluate(real_loader, synthetic_loader),
    #     'nearest_syn_neighbor_distance': NearestSynNeighborDistance().evaluate(real_loader, synthetic_loader),
    #     'close_values_probability': CloseValuesProbability().evaluate(real_loader, synthetic_loader),
    #     'distant_values_probability': DistantValuesProbability().evaluate(real_loader, synthetic_loader)
    # }

    # # Statistical Metrics
    # results['stats'] = {
    #     'jensen_shannon_dist': JensenShannonDist().evaluate(real_loader, synthetic_loader),
    #     'chi_squared_test': ChiSquaredTest().evaluate(real_loader, synthetic_loader),
    #     'feature_correlation': FeatureCorrelation().evaluate(real_loader, synthetic_loader),
    #     'inverse_kl_divergence': InverseKLDivergence().evaluate(real_loader, synthetic_loader),
    #     'ks_test': KSTest().evaluate(real_loader, synthetic_loader),
    #     'max_mean_discrepancy': MaximumMeanDiscrepancy().evaluate(real_loader, synthetic_loader),
    #     'wasserstein_distance': WassersteinDistance().evaluate(real_loader, synthetic_loader),
    #     'prdc': PRDCScore().evaluate(real_loader, synthetic_loader),
    #     'alpha_precision': AlphaPrecision().evaluate(real_loader, synthetic_loader),
    #     'survival_km_distance': SurvivalKMDistance().evaluate(real_loader, synthetic_loader)
    # }

    # # Performance Metrics
    # if target_column:
    #     results['performance'] = {
    #         'linear_model': PerformanceLinear(target_column=target_column).evaluate(real_loader, synthetic_loader),
    #         'mlp': PerformanceMLP(target_column=target_column).evaluate(real_loader, synthetic_loader),
    #         'xgb': PerformanceXGB(target_column=target_column).evaluate(real_loader, synthetic_loader),
    #         'feature_rank_distance': FeatureRankDistance(target_column=target_column).evaluate(real_loader, synthetic_loader)
    #     }
    # else:
    #     results['performance'] = 'Target column required for performance metrics.'

    # # Detection Metrics
    # results['detection'] = {
    #     'detection_xgb': DetectionXGB().evaluate(real_loader, synthetic_loader),
    #     'detection_mlp': DetectionMLP().evaluate(real_loader, synthetic_loader),
    #     'detection_gmm': DetectionGMM().evaluate(real_loader, synthetic_loader),
    #     'detection_linear': DetectionLinear().evaluate(real_loader, synthetic_loader)
    # }

    return results
