from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import pickle


def ordinal_encode_categorical(data, pickle_path: str = 'mappings.pkl'):
    """
    Encodes categorical columns using OrdinalEncoder and saves mappings for reversibility.

    Parameters:
    data (pd.DataFrame): Input DataFrame with categorical columns to encode.
    pickle_path (str): File path to save mappings and column mappings (default is 'mappings.pkl').

    Returns:
    pd.DataFrame: Encoded DataFrame with numerical values replacing categorical columns.
    """
    print('hi!')
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
    encoded_data.to_csv('../data/encoded_data.csv', index=False)
    print('here')
    return encoded_data

def ordinal_decode_categorical(data, pickle_path: str = 'mappings.pkl'):
    """
    Decodes a DataFrame encoded with `ordinal_encode_categorical` using mappings from a pickle file.

    Parameters:
    data (pd.DataFrame): Encoded DataFrame with numerical values.
    pickle_path (str): File path to load mappings (default is 'mappings.pkl').

    Returns:
    pd.DataFrame: Decoded DataFrame with categorical columns restored.
    """
    # Load mappings from the pickle file
    with open(pickle_path, 'rb') as f:
        mappings = pickle.load(f)
    
    decoded_data = data.copy()
    
    # Decode each categorical column
    for col, map_dict in mappings.items():
        categories = map_dict['categories']
        decoded_data[col] = decoded_data[col].map(lambda x: categories[int(x)] if not pd.isna(x) else None)
    
    return decoded_data