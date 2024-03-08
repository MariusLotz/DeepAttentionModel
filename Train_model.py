import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from Functions.Signal_to_Features import signal_to_wavelet_features
from Functions.Helper_Functions import load_from_pickle
from Example_Problems.FordA import FordA_preprocessing


def preprocess_data(file_dest):
    data = load_from_pickle(file_dest)
    inputs = torch.stack([item[0] for item in data])
    labels = torch.stack([item[1] for item in data])
    return inputs, labels


def train_model(model, data_loader, criterion, optimizer, epochs):
    """
    Train a PyTorch model.
    """
    device = torch.device("cuda")
    model = model.to(device)
    model.train()
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU if available
            predictions = model(inputs)
            targets = targets.view(-1, 1)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

            if torch.isnan(loss):
                print(f'Loss became NaN at epoch {epoch + 1}. Training stopped.')
                return model
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

if __name__ == "__main__":
    pass

   
