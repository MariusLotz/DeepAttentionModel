import numpy as np
import random
import pickle
import torch

def simple_cos_freq(category, num_points=128):
    time = np.arange(num_points)
    amplitude = random.uniform(1, 10.0)

    if category == 1:
        frequency = random.uniform(50.0, 100.0)
    else:
        frequency = random.uniform(1, 10.0)
    cos_wave = amplitude * np.cos(2 * np.pi * time * frequency)  # Regular Cosine pattern
    signal = torch.tensor(cos_wave, dtype=torch.float32)
    return signal, torch.tensor(category, dtype=torch.float32)
    

def create_cosine_wave(category, num_points=128):
    """
    Create a synthetic signal with a cosine wave pattern.

    Args:
        category (int): Category of the signal (1 for modified pattern, 0 for regular cosine wave).
        num_points (int, optional): Number of points in the signal. Default is 128.

    Returns:
        torch.Tensor: The generated signal.
        torch.Tensor: The category label.
    """
    time = np.arange(num_points)
    amplitude = random.uniform(1, 10.0)
    frequency = random.uniform(1, 100.0)
    phase = random.uniform(-0.25, 0.25)
    cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequency)  # Regular Cosine pattern
    signal = torch.tensor(cos_wave, dtype=torch.float32)

    if category == 1:
        interval_len = num_points // 16
        interval_time = np.arange(interval_len)
        amplitude = random.uniform(1, 10.0)
        frequency = random.uniform(16, 100.0)
        phase = random.uniform(-0.25, 0.25)
        additional_cos_wave = amplitude * np.cos(2 * np.pi * (interval_time + phase) * frequency)
        random_start = random.randint(0, num_points - interval_len)

        for i in range(interval_len):
            signal[random_start + i] += additional_cos_wave[i]

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
    num_samples = 9

    # Generate training data
    training_data = generate_training_data(num_samples, num_points=12, func=simple_cos_freq)
    for x in training_data:
        print(x)
        print()
  
    # Save training data to a pickle file
    #save_to_pickle(training_data, "test_data_999_simple.pkl")

