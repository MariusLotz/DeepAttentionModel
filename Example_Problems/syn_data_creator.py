# Import fix for Codespace:
import sys
sys.path.append("./")
import numpy as np
import torch
from Helper_Functions import generate_training_data, save_to_pickle

def sin_data_creator(num_points, T, t_1, t_2, f, x, c1, n_1=0.1, n_2=0.1):
    """
    Create a synthetic sin signal

    Args:
        num_points: size of vector representation
        T: Defines time interval [0,T]
        t_1, t_2: Defines time interval for the difference [t_1, t_2]
        f: Frequency of the signal
        x: Difference factor of the signal (x*f)
        c1: class 1 (True or False)
        n_1, n_2: strength of Gaussian noise
    Returns:
        torch.Tensor: The generated signal.
    """

    sample_rate = T / num_points
    max_freq = max(f, f*x)
    if sample_rate < 2 * max_freq:
        raise Exception("Nyquist–Shannon sampling theorem not satisfied")
    
    time = np.linspace(0, T, num_points)
    signal = torch.zeros(num_points)
    for i, t in enumerate(time):
        if t_1 < t <= t_2 and c1:
            noise_2 = np.random.normal(0, n_2)
            signal[i] = np.cos(f*t*x) + noise_2
        else:
            noise_1 = np.random.normal(0, n_1)
            signal[i] = np.cos(f*t) + noise_1

    return signal

def create_sin_ex1(category, num_points=128):
    return sin_data_creator(num_points, 8192, 4096, 8192,1,3, category)

def example():
    # Example usage:
    num_points = 10
    T = 61
    t_1, t_2 = 3, 7
    f = 1
    x = 3
    n_1, n_2 = 0.1, 0.1

    signal = sin_data_creator(num_points, T, t_1, t_2, f, x, n_1, n_2)
    print(signal)

if __name__=="__main__":
    #example()
    training_data = generate_training_data(9999, num_points=128, func=create_sin_ex1)
    # Save training data to a pickle file
    save_to_pickle(training_data, "test_data_9999_sin_ex1.pkl")
