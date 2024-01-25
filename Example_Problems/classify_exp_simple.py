import numpy as np
import random
import pickle
import torch


def simple_cos_freq(category, num_points=16):
    time = np.arange(num_points)
    amplitude = random.uniform(1, 10.0)

    if category == 1:
        frequency = random.uniform(50.0, 100.0)
    else:
        frequency = random.uniform(1, 10.0)
    cos_wave = amplitude * np.cos(2 * np.pi * time * frequency)  # Regular Cosine pattern
    signal = torch.tensor(cos_wave, dtype=torch.float32)
    return signal, torch.tensor(category, dtype=torch.float32)


def generate_training_data(batch_size, num_points, func):
    """
    Generate training data with random features and labels.

    Args:
        batch_size (int): Number of training samples to generate.

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


if __name__ == "__main__":
    # Set the number of training samples
    num_samples = 9999

    # Generate training data
    training_data = generate_training_data(num_samples, num_points=16, func=simple_cos_freq)
 
    #Save training data to a pickle file
    save_to_pickle(training_data, "training_data_9999_simple_16.pkl")