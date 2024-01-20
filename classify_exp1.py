import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def create_cosine_wave(category, num_points=128):
    """
    Create a synthetic signal with a cosine wave pattern.

    Args:
        category (int): Category of the signal (1 for modified pattern, 0 for regular cosine wave).
        num_points (int, optional): Number of points in the signal. Default is 128.

    Returns:
        np.ndarray: The generated signal.
    """
    time = np.arange(num_points)
    amplitude = random.uniform(1, 10.0)
    frequency = random.uniform(1, 100.0)
    phase = random.uniform(-0.25, 0.25)
    cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequency)  # Regular Cosine pattern
    signal = cos_wave

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

    return signal


def generate_training_data(num_samples):
    """
    Generate training data with random features and labels.

    Args:
        num_samples (int): Number of training samples to generate.

    Returns:
        list: List of tuples, each containing a feature vector and its corresponding label.
    """
    training_data = []

    for _ in range(num_samples):
        # Generate random features (for example, a vector of 3 integers)
        features = [random.randint(1, 10) for _ in range(3)]

        # Generate a random label (for example, a binary label)
        label = random.choice([0, 1])

        # Append the feature-label pair to the training data
        training_data.append((features, label))

    return training_data

def save_to_pickle(data, filename):
    """
    Save data to a pickle file.

    Args:
        data: Data to be saved.
        filename (str): Name of the pickle file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    # Set the number of training samples
    num_samples = 1000

    # Generate training data
    training_data = generate_training_data(num_samples)

    # Save training data to a pickle file
    save_to_pickle(training_data, "training_data.pkl")
