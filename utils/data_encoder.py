import pandas as pd
import numpy as np
from category_encoders.hashing import HashingEncoder

def hash_encode_categorical(data, n_components: int = 10) -> pd.DataFrame:
    """
    Encodes categorical columns in the given DataFrame or NumPy array using HashingEncoder.
    
    Parameters:
    data (pd.DataFrame or np.ndarray): The input data.
    n_components (int): The number of output components for the hashing encoder (default is 10).

    Returns:
    pd.DataFrame: A DataFrame with all numerical values, suitable for DBSCAN.
    """
    # If the input is a NumPy array, convert it to a DataFrame
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame or a NumPy array.")
    
    # Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    encoder = HashingEncoder(cols=categorical_columns, n_components=n_components)
    
    # Transform and return the encoded DataFrame
    df_encoded = encoder.fit_transform(data)
    
    return df_encoded.astype(float)

# Example usage:
# data_array = np.array([['A', 1], ['B', 2], ['C', 3]])
# encoded_df = hash_encode_categorical(data_array)
# print(encoded_df)