import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from Model import Feature2LBinaryClassifier as model
import matplotlib.pyplot as plt
from Signal_to_Features import signal_to_wavelet_features


def load_from_pickle(filename):
    """Load data from pickle file"""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def preprocess_data(file_dest):
    """Processing steps for data"""
    # Prepare data for training
    data = load_from_pickle(file_dest)

    # Convert to PyTorch tensors
    inputs = torch.stack([item[0] for item in data])
    labels = torch.stack([item[1] for item in data])
    
    return inputs, labels


def train_model(model, data_loader, criterion, optimizer, epochs=1000):
    model.train()
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        for inputs, targets in data_loader:
            predictions = model(inputs)
            

            targets = targets.view(-1, 1)
            #print(targets.shape)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

            if torch.isnan(loss):
                print(f'Loss became NaN at epoch {epoch + 1}. Training stopped.')
                return model
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        #if (epoch + 1) % 100 == 0:
            

    return model


def train(number):
    my_model = model(128, signal_to_wavelet_features)
    model_path = f"model_pre_training_{number}"  # Updated model_path with f-string
    # torch.save(my_model.state_dict(), model_path)
    my_criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    my_optimizer = optim.Adam(my_model.parameters(), lr=0.01)
    my_batch_size = 128
    my_num_epochs = 1000  

    inputs, labels = preprocess_data("Example_Problems/training_data_99999_128.pkl")

    my_dataset = TensorDataset(inputs, 
                               labels)
    my_data_loader = DataLoader(my_dataset, batch_size=my_batch_size, shuffle=True)

    trained_model = train_model(my_model, my_data_loader, my_criterion, my_optimizer, epochs=my_num_epochs)

    model_path = f"model_post_training_{number}"  # Updated model_path with f-string
    torch.save(trained_model.state_dict(), model_path)

if __name__ == "__main__":
    train(1)
