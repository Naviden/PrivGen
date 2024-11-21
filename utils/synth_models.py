import pandas as pd
from synthcity.plugins import Plugins
from synthcity.utils.datasets import load_dataset

def get_synthetic_data(dataset: pd.DataFrame, model_name: str, num_samples: int):
    """
    Generate synthetic data using the specified model from SynthCity.

    Parameters:
        dataset (pd.DataFrame): The original dataset.
        model_name (str): The name of the synthetic data generation model. Options: 'AdsGAN', 'PATEGAN', 'PrivBayes', 'DP-GAN', 'DECAF'.
        num_samples (int): The number of synthetic samples to generate.

    Returns:
        pd.DataFrame: The generated synthetic data.
    """
    # Check if the selected model is available
    available_models = ['AdsGAN', 'PATEGAN', 'PrivBayes', 'DP-GAN', 'DECAF']
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from {available_models}.")

    # Load the corresponding plugin for the chosen model
    plugin = Plugins().get(model_name)

    # Train the model on the provided dataset
    plugin.fit(dataset)

    # Generate synthetic data
    synthetic_data = plugin.generate(num_samples)

    return synthetic_data

# Main function for user interaction
if __name__ == "__main__":
    # Load your dataset
    print("Loading dataset...")
    # Replace with your actual dataset
    dataset = load_dataset("adult")  # Example dataset from SynthCity utils
    print("Dataset loaded.")

    # Display available models
    models = ['AdsGAN', 'PATEGAN', 'PrivBayes', 'DP-GAN', 'DECAF']
    print("Available models for synthetic data generation:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    # Get user input for model selection
    model_index = int(input(f"Select a model by entering a number (1-{len(models)}): ")) - 1
    if model_index < 0 or model_index >= len(models):
        raise ValueError("Invalid model selection.")
    selected_model = models[model_index]

    # Get user input for the number of synthetic samples
    num_samples = int(input("Enter the number of synthetic samples to generate: "))

    # Generate synthetic data
    print(f"Generating synthetic data using {selected_model}...")
    synthetic_data = get_synthetic_data(dataset, selected_model, num_samples)

    # Display the synthetic data
    print("Synthetic data generated successfully:")
    print(synthetic_data.head())

    # Save to CSV (optional)
    save_option = input("Do you want to save the synthetic data to a CSV file? (yes/no): ").strip().lower()
    if save_option == "yes":
        file_name = input("Enter the file name (e.g., synthetic_data.csv): ").strip()
        synthetic_data.to_csv(file_name, index=False)
        print(f"Synthetic data saved to {file_name}.")