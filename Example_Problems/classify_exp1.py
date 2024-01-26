import numpy as np
import random
import torch
from Helper_Functions import save_to_pickle, generate_training_data

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


if __name__ == "__main__":
    # Set the number of training samples
    num_samples = 9

    # Generate training data
    training_data = generate_training_data(num_samples, num_points=12, func=create_cosine_wave)
   

    # Save training data to a pickle file
    # save_to_pickle(training_data, "training_data_99999_1.pkl")

