import random
import pickle
import pandas as pd
import torch


def data_table_to_tensors(file_path, file_type, dataset_trafo):
    """
    Read a dataset file into Pandas DataFrame, apply modifications, and convert to PyTorch tensors.

    Parameters:
    - file_path (str): Path to the dataset file.
    - file_type (str): Type of the dataset file.
    - dataset_trafo (callable): A function to transform the Pandas DataFrame.

    Returns:
    - tuple: A tuple containing PyTorch tensors for classes and time series.
    """
    # Read the dataset file into a Pandas DataFrame
    if file_type.lower() == 'csv':
        df = pd.read_csv(file_path, sep=',', header=None)  
    elif file_type.lower() == 'tsv':
        df = pd.read_csv(file_path, sep='\t', header=None)
    else:
        raise ValueError("Unsupported file type. Supported types: 'csv', 'tsv'")

    # Apply dataset-specific transformations to the DataFrame
    class_column, time_series_columns = dataset_trafo(df)

    # Convert NumPy arrays to tensors
    classes_tensor = torch.tensor(class_column, dtype=torch.float) 
    time_series_tensor = torch.tensor(time_series_columns, dtype=torch.float)

    return classes_tensor, time_series_tensor


def generate_training_data(batch_size, num_points, func):
    """
    Generate training data with random features and labels.

    Args:
        batch_size (int): Number of training samples to generate.
        num_points (int): Number of points for the generated data.
        func (callable): A function to generate data based on the number of points.

    Returns:
        list: List of tuples, each containing a feature tensor and its corresponding label tensor.
    """
    data = [(func(random.choice([0, 1]), num_points)) for _ in range(batch_size)]
    return data


def save_to_pickle(data, filename):
    """
    Save data to a pickle file.

    Args:
        data (list): Data to be saved.
        filename (str): Name of the pickle file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_from_pickle(filename):
    """
    Load data from a pickle file.

    Args:
        filename (str): The path to the pickle file.

    Returns:
        data: The loaded data.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def get_xth_pickle(filename, xth):
    """
    Get the x-th element from a pickle file.

    Args:
        filename (str): The path to the pickle file.
        xth (int): The index of the element to retrieve.

    Returns:
        The x-th element from the pickle file.
    """
    data = load_from_pickle(filename)
    return data[xth]


def get_first_x_pickle(filename, x):
    """
    Get the first x elements from a pickle file.

    Args:
        filename (str): The path to the pickle file.
        x (int): The number of elements to retrieve.

    Returns:
        List of the first x elements from the pickle file.
    """
    data = load_from_pickle(filename)
    return data[:x]


if __name__ == "__main__":
    print(get_first_x_pickle("Example_Problems/training_data_9999_simple_128.pkl", 4))
    

