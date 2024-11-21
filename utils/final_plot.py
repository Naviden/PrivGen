import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_data_points():
    """
    Loads three CSV files, reduces dimensions to 2 if necessary, plots all points with specific markers 
    and colors for each file, and displays a scatter plot.
    
    Files:
        - final_cleaned.csv: Blue circles for non-outliers.
        - outliers_by_DBSCAN.csv: Red balls for DBSCAN outliers.
        - outliers_by_distance.csv: Red crosses for distance-based outliers.
    
    All files must be in the '../data/' folder.
    """
    def preprocess_data(file_path):
        """
        Load the CSV file, drop 'cluster' and 'distance' columns if they exist, 
        and reduce dimensions to 2 if necessary.
        
        Parameters:
            file_path (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: Processed DataFrame with 2 columns for plotting.
        """
        df = pd.read_csv(file_path)
        # Drop 'cluster' and 'distance' columns if they exist
        df = df.drop(columns=['cluster', 'distance'], errors='ignore')
        
        # Reduce to 2 dimensions if more than 2 columns
        if df.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(df)
            df = pd.DataFrame(reduced_data, columns=['feature_0', 'feature_1'])
        else:
            # Rename columns for consistency
            df.columns = ['feature_0', 'feature_1']
        
        return df
    
    # Preprocess the data from each file
    final_cleaned = preprocess_data('../data/final_cleaned.csv')
    outliers_dbscan = preprocess_data('../data/outliers_by_DBSCAN.csv')
    outliers_distance = preprocess_data('../data/outliers_by_distance.csv')
    
    # Create the scatter plot
    plt.figure(figsize=(10, 7))
    
    # Plot data points
    plt.scatter(final_cleaned['feature_0'], final_cleaned['feature_1'], 
                color='blue', marker='o', label='Final Cleaned', alpha=0.7)
    plt.scatter(outliers_dbscan['feature_0'], outliers_dbscan['feature_1'], 
                color='red', marker='o', label='Outliers by DBSCAN', alpha=0.7)
    plt.scatter(outliers_distance['feature_0'], outliers_distance['feature_1'], 
                color='red', marker='x', label='Outliers by Distance', alpha=0.7)
    
    # Plot details
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('Scatter Plot of Data Points')
    plt.legend(loc='best')
    
    # Display the plot
    plt.show()