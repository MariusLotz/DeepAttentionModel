import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Models import Feature2LBinaryClassifier, SimpleBinaryClassifier, RawSimpleBinaryClassifier
from datetime import datetime
from Signal_to_Features import signal_to_wavelet_features
from Helper_Functions import load_from_pickle
from Example_Problems.FordA import FordA_preprocessing


def preprocess_data(file_dest):
    """
    Process data loaded from a pickle file.

    Parameters:
    - file_dest (str): The path to the pickle file.

    Returns:
    - inputs (torch.Tensor): PyTorch tensor containing input data.
    - labels (torch.Tensor): PyTorch tensor containing labels.
    """
    # Prepare data for training
    data = load_from_pickle(file_dest)

    # Convert to PyTorch tensors
    inputs = torch.stack([item[0] for item in data])
    labels = torch.stack([item[1] for item in data])
    return inputs, labels


def train_model(model, data_loader, criterion, optimizer, epochs):
    """
    Train a PyTorch model.

    Parameters:
    - model (nn.Module): The PyTorch model to be trained.
    - data_loader (DataLoader): DataLoader for loading training data.
    - criterion: The loss criterion used for training.
    - optimizer: The optimizer used for updating model parameters.
    - epochs (int): Number of training epochs.

    Returns:
    - model (nn.Module): The trained PyTorch model.
    """
    model.train()
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        for inputs, targets in data_loader:
            predictions = model(inputs)
            targets = targets.view(-1, 1)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

            if torch.isnan(loss):
                print(f'Loss became NaN at epoch {epoch + 1}. Training stopped.')
                return model
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model


def train_model_with_params(model_class, *param, losscriterion, optimizer, batchsize, num_epochs, inputs, outputs):
    """
    Train a PyTorch model with specified parameters.

    Parameters:
    - model_class (nn.Module): The PyTorch model class.
    - signal_size (int): The size of the input signal for model initialization.
    - losscriterion: The loss criterion used for training.
    - optimizer: The optimizer used for updating model parameters.
    - batchsize (int): Batch size for DataLoader.
    - num_epochs (int): Number of training epochs.
    - inputs (torch.Tensor): PyTorch tensor containing input data.
    - outputs (torch.Tensor): PyTorch tensor containing labels.
    """
    # Get the current date and time
    current_date_time = datetime.now()

    # Format the date as a string
    current_date_string = current_date_time.strftime("%Y-%m-%d")

    my_model_untrained = model_class(*param)
    model_path_untrained = f"Models/{model_class.__name__}_{current_date_string}_untrained"
    torch.save(my_model_untrained, model_path_untrained)

    my_criterion = losscriterion()
    my_optimizer = optimizer(my_model_untrained.parameters())
    
    my_dataset = TensorDataset(inputs, outputs)
    my_data_loader = DataLoader(my_dataset, batch_size=batchsize, shuffle=True)

    trained_model = train_model(my_model_untrained, my_data_loader, my_criterion, my_optimizer, epochs=num_epochs)

    model_path_trained = f"Models/{model_class.__name__}_{current_date_string}_trained"
    torch.save(trained_model, model_path_trained)


def train_RawSimpleBinaryClassifier():
    """
    Train the SimpleBinaryClassifier model with predefined parameters.
    """
    # Load training data:
    labels, inputs = FordA_preprocessing()
    signalsize = inputs.size(1)
    # Create and save model pretrained and posttrained
    train_model_with_params(RawSimpleBinaryClassifier, signalsize, losscriterion=nn.BCELoss, 
                            optimizer=optim.Adam, batchsize=512, num_epochs=100, inputs=inputs, outputs=labels)


def train_SimpleBinaryClassifier():
    """
    Train the SimpleBinaryClassifier model with predefined parameters.
    """
    # Load training data:
    labels, inputs = FordA_preprocessing()
    signalsize = inputs.size(1)
    # Create and save model pretrained and posttrained
    train_model_with_params(SimpleBinaryClassifier, signalsize, losscriterion=nn.BCELoss, 
                            optimizer=optim.Adam, batchsize=512, num_epochs=100, inputs=inputs, outputs=labels)


def train_Feature2LBinaryClassifier():
    """
    Train the SimpleBinaryClassifier model with predefined parameters.
    """
    feature_function = signal_to_wavelet_features
    # Load training data:
    labels, inputs = FordA_preprocessing()
    signalsize = inputs.size(1)

    # Create and save model pretrained and posttrained
    train_model_with_params(Feature2LBinaryClassifier, signalsize, feature_function, losscriterion=nn.BCELoss, 
                            optimizer=optim.Adam, batchsize=512, num_epochs=100, inputs=inputs, outputs=labels)

if __name__ == "__main__":
    #train_RawSimpleBinaryClassifier()
    #train_SimpleBinaryClassifier()
    train_Feature2LBinaryClassifier()
