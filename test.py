# from clustering_search import run_clustering_and_plot

# results = run_clustering_and_plot()
# results.to_excel('cane.xlsx', index=False)

# import pandas as pd

# def find_top_records(dataframe, num_clusters, num_outliers, top_n=5):
#     """
#     Filters the dataframe to return top N rows with the specified number of clusters
#     and the closest number of outliers to the specified value.

#     Parameters:
#         dataframe (pd.DataFrame): The input dataframe.
#         num_clusters (int): The specified number of clusters.
#         num_outliers (int): The target number of outliers.
#         top_n (int): Number of top records to return (default is 5).

#     Returns:
#         pd.DataFrame: Filtered dataframe with the top N closest rows.
#     """
#     # Filter rows with the specified number of clusters
#     filtered = dataframe[dataframe['num_clusters'] == num_clusters]
    
#     # Calculate the absolute difference in 'num_outliers'
#     filtered['outlier_difference'] = (filtered['num_outliers'] - num_outliers).abs()
    
#     # Sort by the absolute difference and return the top N rows
#     top_records = filtered.sort_values('outlier_difference').head(top_n)
    
#     return top_records.drop(columns=['outlier_difference'])



# # Usage example (assuming your DataFrame is named df):
# print(find_top_records(results, num_clusters=3, num_outliers=10))


