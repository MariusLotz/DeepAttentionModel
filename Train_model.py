import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define your model architecture here

    def forward(self, x):
        # Define the forward pass of your model
        return x

def train_model(model, data_loader, criterion, optimizer, epochs=1000):
    model.train()
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        for inputs, targets in data_loader:
            predictions = model(inputs)

            # No need for sigmoid here, as BCEWithLogitsLoss combines sigmoid and binary cross-entropy
            targets = targets.view(-1, 1)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

            if torch.isnan(loss):
                print(f'Loss became NaN at epoch {epoch + 1}. Training stopped.')
                return model

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

def train(number):
    my_model = Model()
    model_path = f"model_pre_training_{number}"  # Updated model_path with f-string
    # torch.save(my_model.state_dict(), model_path)
    my_criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    my_optimizer = optim.Adam(my_model.parameters(), lr=0.01)
    my_batch_size = 128
    my_num_epochs = 400  # Add the number of epochs here

    with open("Data/10000_trainingsample_4dim_2.pkl", 'rb') as file:
        my_sample = pickle.load(file)

    my_dataset = TensorDataset(torch.tensor(my_sample[0], dtype=torch.float32), 
                               torch.tensor(my_sample[1], dtype=torch.float32))
    my_data_loader = DataLoader(my_dataset, batch_size=my_batch_size, shuffle=True)

    trained_model = train_model(my_model, my_data_loader, my_criterion, my_optimizer, epochs=my_num_epochs)

    model_path = f"model_post_training_{number}"  # Updated model_path with f-string
    torch.save(trained_model.state_dict(), model_path)

if __name__ == "__main__":
    train(2)
