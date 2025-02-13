from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import pickle


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

def ordinal_decode_categorical(dataset_name):
    """
    Decodes a DataFrame encoded with `ordinal_encode_categorical` using mappings from a pickle file.

    Parameters:
    data (pd.DataFrame): Encoded DataFrame with numerical values.
    pickle_path (str): File path to load mappings (default is 'mappings.pkl').

    Returns:
    pd.DataFrame: Decoded DataFrame with categorical columns restored.
    """
    pickle_path = f'{dataset_name}_mappings.pkl'
    # Load mappings from the pickle file
    with open(pickle_path, 'rb') as f:
        mappings = pickle.load(f)
    

    data = pd.read_csv(f'../data/{dataset_name}_5_final_cleaned.csv')
    data.drop(['cluster', 'distance'], axis=1, inplace=True)
    data.to_csv(f'../data/{dataset_name}_7_before_decoding.csv')

    decoded_data = data.copy()
    
    # Decode each categorical column
    for col, map_dict in mappings.items():
        categories = map_dict['categories']
        decoded_data[col] = decoded_data[col].map(lambda x: categories[int(x)] if not pd.isna(x) else None)
    decoded_data.to_csv(f'../data/{dataset_name}_8_decoded_data.csv', index=False)
    
    return decoded_data




def one_hot_encode_categorical(data, dataset_name, save=True):
    """
    Encodes categorical columns using One-Hot Encoding and saves mappings for reversibility.

    Parameters:
    data (pd.DataFrame): Input DataFrame with categorical columns to encode.
    dataset_name (str): Unique identifier for saving mappings.
    save (bool): Whether to save the encoded data to a CSV file.

    Returns:
    pd.DataFrame: Encoded DataFrame with one-hot encoded categorical columns.
    """
    pickle_path = f'{dataset_name}_mappings.pkl'

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Apply encoding only to categorical columns
    encoded_array = encoder.fit_transform(data[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    
    # Convert back to DataFrame
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=data.index)
    
    # Drop original categorical columns and merge the one-hot encoded ones
    encoded_data = data.drop(columns=categorical_columns).reset_index(drop=True)
    encoded_data = pd.concat([encoded_data, encoded_df.reset_index(drop=True)], axis=1)
    
    # Store mappings
    mappings = {'categorical_columns': list(categorical_columns), 'encoder': encoder}

    # Save mappings to a pickle file
    with open(pickle_path, 'wb') as f:
        pickle.dump(mappings, f)

    if save:
        encoded_data.to_csv(f'../data/{dataset_name}_1_encoded_data.csv', index=False)
    
    return encoded_data

def one_hot_decode_categorical(dataset_name):
    """
    Decodes a DataFrame encoded with `one_hot_encode_categorical` using mappings from a pickle file.

    Parameters:
    dataset_name (str): Unique identifier for loading mappings and decoding data.

    Returns:
    pd.DataFrame: Decoded DataFrame with categorical columns restored.
    """
    pickle_path = f'{dataset_name}_mappings.pkl'

    # Load mappings from the pickle file
    with open(pickle_path, 'rb') as f:
        mappings = pickle.load(f)

    data = pd.read_csv(f'../data/{dataset_name}_5_final_cleaned.csv')
    data.drop(['cluster', 'distance'], axis=1, inplace=True)
    data.to_csv(f'../data/{dataset_name}_7_before_decoding.csv', index=False)

    categorical_columns = mappings['categorical_columns']
    encoder = mappings['encoder']

    # Extract encoded feature names
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    
    # Select only the one-hot encoded part
    encoded_data = data[encoded_feature_names]
    
    # Decode back to original categories
    decoded_array = encoder.inverse_transform(encoded_data)
    decoded_df = pd.DataFrame(decoded_array, columns=categorical_columns, index=data.index)

    # Drop one-hot encoded columns and merge decoded categorical columns
    decoded_data = data.drop(columns=encoded_feature_names).reset_index(drop=True)
    decoded_data = pd.concat([decoded_data, decoded_df.reset_index(drop=True)], axis=1)

    decoded_data.to_csv(f'../data/{dataset_name}_8_decoded_data.csv', index=False)
    
    return decoded_data