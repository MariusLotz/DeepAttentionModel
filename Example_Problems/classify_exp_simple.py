import numpy as np
import random
import torch
from Helper_Functions import save_to_pickle, generate_training_data
# Import fix for Codespace:
import sys
sys.path.append("./")


def simple_cos_freq(category, num_points=128):
    time = np.arange(num_points)
    #amplitude = random.uniform(1, 1.0)
    amplitude = 1

    if category == 1:
        frequency = random.uniform(0.9, 1.0)
    else:
        frequency = random.uniform(0.1, 0.2)
    cos_wave = amplitude * np.cos(2 * np.pi * time * frequency)  # Regular Cosine pattern
    signal = torch.tensor(cos_wave, dtype=torch.float32)
    return signal, torch.tensor(category, dtype=torch.float32)


if __name__ == "__main__":
    # Set the number of training samples
    num_samples = 9999

    # Generate training data
    training_data = generate_training_data(num_samples, num_points=128, func=simple_cos_freq)
 
    #Save training data to a pickle file
    save_to_pickle(training_data, "Example_Problems/Data/training_data_9999_simple_128.pkl")